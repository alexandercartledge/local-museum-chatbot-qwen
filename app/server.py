import os
import pickle
import re
from collections import defaultdict
from typing import List, Optional
import json
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------

INDEX_DIR = os.getenv("INDEX_DIR", "./index")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
ROOM_MIN_SIM = float(os.getenv("ROOM_MIN_SIM", "0.40"))  # tighter by default
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "8000"))
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "10"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "3000"))

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q4_0")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Optional critic pass (self-check) – disabled by default
ENABLE_CRITIC = os.getenv("ENABLE_CRITIC", "0") == "1"
CRITIC_MODEL = os.getenv("CRITIC_MODEL", LLM_MODEL)


# common operational queries we will always answer with a fixed message
OFFTOPIC_RE = re.compile(
    r"\b("
    r"orari?|apertur[ae]|chiusur[ae]|prezz[oi]|costi?|bigliett[io]|ticket|"
    r"parchegg|come\s+arrivare|indirizz|telefono|email|contatti?|prenotaz|booking|"
    r"shop|negozio|caff[eè]|bar|ristorante|toilett|bagni|"
    r"opening\s+hours?|opening\s+times?|closing\s+time|schedule|timetable|"
    r"admission|entrance|entry|price|prices|cost|costs|fee|fees|"
    r"library|biblioteca|bookshop|book\s*store|"
    r"foto|fotograf|photo|pictures?|selfie|video|filmare|riprendere|camera"
    r")\b",
    re.I,
)



# Synthetic room id for general museum information (hours, tickets, contacts...)
INFO_ROOM_ID = "GDA-Info-Museo"


MUSEUM_INFO_IT = """
Museo delle Genti d’Abruzzo – Informazioni per la visita

INDIRIZZO
- Museo delle Genti d’Abruzzo
  Via delle Caserme 24, 65127 Pescara (PE), Italia
- Telefono centralino: +39 085 451 0026
- Email generale: museo@gentidabruzzo.it
- Email didattica / scuole: didattica@gentidabruzzo.it

ORARI DI APERTURA (Museo delle Genti d’Abruzzo – dal 22/09/2025)
- Lunedì–Venerdì: 09:00–13:00
- Sabato–Domenica: 16:00–20:00
- Chiusure: 1 gennaio, Pasqua, 1 novembre, 25 e 26 dicembre.
(L’orario può variare: per sicurezza controlla sempre il sito ufficiale.)

MUSEO BASILIO CASCELLA (stessa fondazione)
- Lunedì–Giovedì: 09:00–13:00 (solo su prenotazione entro 3 giorni)
- Venerdì: 09:00–13:00
- Sabato–Domenica: 16:00–20:00
- Prenotazioni: +39 085 451 0026 int. 1 – museo@gentidabruzzo.it / info@museocascella.it
- Chiusure festive come il Museo delle Genti d’Abruzzo.

ORARI NEGOZIO / BOOKSHOP
- Aperto negli stessi orari del museo.
- Aperture straordinarie su richiesta per piccoli gruppi (min. 6 persone, prenotazione almeno 48 ore prima).

ORARI BIBLIOTECA (servizi bibliotecari e sala lettura)
- Lunedì: 09:00–13:00 e 15:30–18:30
- Martedì: 09:00–13:00
- Mercoledì: 15:30–18:30
- Giovedì: 15:30–20:30 (dalle 18:30 solo sala lettura)
- Venerdì: 09:00–13:00
- Info e prenotazioni: tel. +39 085 451 1562 (int. 5) – biblioteca@gentidabruzzo.it

BIGLIETTI – MUSEO DELLE GENTI D’ABRUZZO
- Intero adulti: 8 €
- Ridotto over 65: 5 €
- Ridotto under 18: 5 €

BIGLIETTO CUMULATIVO (Museo delle Genti d’Abruzzo + Museo Civico “B. Cascella”)
- Intero adulti: 12 €
- Ridotto (under 18 e over 65): 8 €

INGRESSO GRATUITO
- Bambini fino a 3 anni
- Persone con disabilità
- Soci ASTRA – Amici del Museo delle Genti d’Abruzzo
- Soci Archeoclub
- Soci ICOM
- Donatori AVIS e FIDAS (gratuità per la mostra permanente sul Risorgimento in Abruzzo)

RIDUZIONI (esempi principali)
- Convenzioni: Abruzzo B&B, FAI, dipendenti Questura di Pescara, dipendenti Amministrazione Penitenziaria
- Studenti universitari
- Soci VIVIPARCHI
- Gruppi (almeno 15 persone)
- Soci Touring Club: sconto 50%
- Card Consorzio Turistico Montesilvano
(Le condizioni possono cambiare: per dettagli aggiornati vedere la sezione “Tariffe e Informazioni” del sito.)

COME ARRIVARE
- Indirizzo: Via delle Caserme 24, Pescara.
- Autobus urbani: linee 3, 10, 21, 38 (fermate in zona Porta Nuova).
- Treno: stazione Pescara Porta Nuova a breve distanza.
- Auto: possibilità di parcheggio nei pressi del museo (vedi mappa collegata sul sito).
- Mobilità sostenibile: disponibilità di monopattini in sharing (es. Helbiz) nella zona.

SERVIZI AL PUBBLICO
- Ristorante / Caffè Letterario: pranzi, cene, catering, banchetti.
- Biblioteca: supporto alla ricerca sul territorio abruzzese; consultazione aperta a tutti su prenotazione.
- Punto vendita: pubblicazioni del museo, libri per bambini, cartoline, gadget, giochi, cancelleria, accessori e altro.
- Visite guidate per gruppi su prenotazione (contattare il museo per info su costi e lingue disponibili).
- Accessibilità: informazioni specifiche nella sezione “Accessibilità” del sito.

MOSTRE ED EVENTI
- Mostre temporanee e iniziative culturali sono elencate e aggiornate nelle sezioni “Mostre” ed “Eventi” del sito gentidabruzzo.com.
- Per laboratori didattici, attività per scuole, famiglie e adulti, consultare la sezione “Servizi educativi”.

NOTE
- Le tariffe, gli orari e le convenzioni possono subire modifiche. In caso di dubbio, fai sempre riferimento alle informazioni più recenti pubblicate sul sito ufficiale del museo.
- si puo fare video e foto nel museo
"""

MUSEUM_INFO_EN = """
Genti d’Abruzzo Museum – Visitor information

ADDRESS
- Genti d’Abruzzo Museum
  Via delle Caserme 24, 65127 Pescara (PE), Italy
- Main phone: +39 085 451 0026
- General email: museo@gentidabruzzo.it
- Education / schools: didattica@gentidabruzzo.it

OPENING HOURS (Genti d’Abruzzo Museum – from 22 Sept 2025)
- Monday–Friday: 09:00–13:00
- Saturday–Sunday: 16:00–20:00
- Closed on: 1 January, Easter Sunday, 1 November, 25 and 26 December.
(Times may change; always check the official website before your visit.)

BASILIO CASCELLA MUSEUM (same foundation)
- Monday–Thursday: 09:00–13:00 (only by reservation at least 3 days in advance)
- Friday: 09:00–13:00
- Saturday–Sunday: 16:00–20:00
- Bookings: +39 085 451 0026 ext. 1 – museo@gentidabruzzo.it / info@museocascella.it
- Closed on the same main holidays as Genti d’Abruzzo.

SHOP / BOOKSHOP
- Open during museum hours.
- Special openings for small groups on request (minimum 6 people, booking at least 48 hours in advance).

LIBRARY HOURS
- Monday: 09:00–13:00 and 15:30–18:30
- Tuesday: 09:00–13:00
- Wednesday: 15:30–18:30
- Thursday: 15:30–20:30 (reading room only after 18:30)
- Friday: 09:00–13:00
- Info and bookings: +39 085 451 1562 (ext. 5) – biblioteca@gentidabruzzo.it

TICKETS – GENTI D’ABRUZZO MUSEUM
- Adult: 8 €
- Reduced 65+: 5 €
- Reduced under 18: 5 €

COMBINED TICKET (Genti d’Abruzzo Museum + “B. Cascella” Civic Museum)
- Adult: 12 €
- Reduced (under 18 and over 65): 8 €

FREE ADMISSION
- Children up to 3 years
- Visitors with disabilities
- Members of: ASTRA – Friends of the Genti d’Abruzzo Museum, Archeoclub, ICOM
- AVIS and FIDAS blood donors (free entry to the permanent exhibition on the Risorgimento in Abruzzo)

DISCOUNTS (main examples)
- Partner rates for: Abruzzo B&B, FAI, staff of Pescara Police HQ, staff of the Prison Administration
- University students
- VIVIPARCHI members
- Groups of at least 15 people
- Touring Club members: 50% discount
- Card holders of Consorzio Turistico Montesilvano
(Conditions and partners may change; for exact current rules see the “Tariffe e Informazioni” page.)

HOW TO GET THERE
- Address: Via delle Caserme 24, Pescara.
- City buses: lines 3, 10, 21, 38 (stops around Porta Nuova).
- Train: Pescara Porta Nuova station within walking distance.
- Car: parking available in the streets near the museum (see linked map on the website).
- Sustainable mobility: electric scooters in sharing (e.g. Helbiz) are usually available in the area.

VISITOR SERVICES
- Restaurant / Literary Café: lunches, dinners, catering and banquets.
- Library: research-oriented library on Abruzzo history and culture, open to the public by reservation.
- Bookshop: museum publications, children’s books, postcards, gadgets, stationery, board games, accessories and more.
- Guided tours for groups available on reservation (contact the museum for prices and available languages).
- Accessibility information is provided in the “Accessibilità / Accessibility” section of the website.

EXHIBITIONS AND EVENTS
- Temporary exhibitions and cultural events are listed and regularly updated in the “Mostre” (Exhibitions) and “Eventi” (Events) sections at gentidabruzzo.com.
- Educational activities and workshops for schools, families and adults are described under “Servizi educativi”.

NOTES
- Opening hours, prices and discounts can change. When in doubt, rely on the latest information published on the museum’s official website.
"""

# -------------------------------------------------------------
# Load room data from meta.pkl and build room-level embeddings
# -------------------------------------------------------------

with open(os.path.join(INDEX_DIR, "meta.pkl"), "rb") as f:
    META = pickle.load(f)["records"]

embed_model = SentenceTransformer(EMBED_MODEL)

ROOM_IDS: List[str] = []
ROOM_DATA: dict = {}

agg_it = defaultdict(list)
agg_en = defaultdict(list)
room_heading = {}
room_url = {}

for rec in META:
    # We only care about room-level records for this architecture
    if rec.get("scope_type") != "room":
        continue

    rid = rec["scope_id"]
    text_it = (rec.get("text_it") or "").strip()
    text_en = (rec.get("text_en") or "").strip()

    if text_it:
        agg_it[rid].append(text_it)
    if text_en:
        agg_en[rid].append(text_en)

    if rid not in room_heading and rec.get("heading"):
        room_heading[rid] = rec["heading"]
    if rid not in room_url and rec.get("url"):
        room_url[rid] = rec["url"]

ROOM_IDS = sorted(agg_it.keys())
for rid in ROOM_IDS:
    ROOM_DATA[rid] = {
        "room_id": rid,
        "heading": room_heading.get(rid, f"Room {rid}"),
        "url": room_url.get(rid, ""),
        "text_it": " ".join(agg_it[rid]),
        "text_en": " ".join(agg_en.get(rid, [])),
    }

# -------------------------------------------------------------
# Add synthetic "museum info" room using the hard-coded texts
# -------------------------------------------------------------
ROOM_DATA[INFO_ROOM_ID] = {
    "room_id": INFO_ROOM_ID,
    "heading": "Informazioni Museo / Museum info",
    "url": "",
    "text_it": MUSEUM_INFO_IT,
    "text_en": MUSEUM_INFO_EN,
}

if INFO_ROOM_ID not in ROOM_IDS:
    ROOM_IDS.append(INFO_ROOM_ID)

ROOM_IDS = sorted(ROOM_IDS)


# -------------------------------------------------------------
# Custom per-room descriptions for the classifier
# (keys MUST match scope_id values in chunks.csv)
# -------------------------------------------------------------

CUSTOM_ROOM_DESCRIPTIONS = {

    "GDA-Sala-1": (
        "Chronological overview of Abruzzo prehistory and protohistory, from the earliest Homo erectus and Neanderthals to the arrival of Homo sapiens, the Mesolithic crisis, the Neolithic agricultural revolution, and the later Copper, Bronze, and Iron Ages in the region. This room is ONLY about very ancient periods before the Roman Empire and before medieval or modern peasants: early humans, stone tools, the first Neolithic farmers and herders, the development of metallurgy, and the emergence of Italic peoples before and during Roman conquest.",
        "Contains information on Paleolithic hunting, Mesolithic small-game strategies, Neolithic crops such as wheat, barley, and farro, the invention of impressed pottery, the building of the first huts and villages, and the spread of metal weapons and tools among warrior societies. Use this room for ANY question about the diet or daily life of Paleolithic, Mesolithic, Neolithic, Bronze Age or Iron Age people, Italic tribes, the Social War, or the collapse of Roman order after barbarian invasions – NOT the later contadini or 19th–20th century farmers."
    ),

    "GDA-Sala-2": (
        "Thematic room dedicated to the sacred use of caves in Abruzzo, showing how natural grottoes became places of worship, ritual pits, and stone circles linked to the cult of Mother Earth from the Neolithic onward. It explains the difference between caves as sanctuaries for offerings and prayers versus open-air villages for everyday life, emphasizing religious practices rather than ordinary dwelling, farming, or domestic routines.",
        "Includes the Grotta dei Piccioni with ritual deposits and child sacrifices, ex-votos in ceramic, stone, and bone, and the long continuity of pagan rites adapted into Christian worship by hermit monks and saints such as Saint Michael the Archangel. It focuses on cave sanctuaries, healing practices connected to rock and water, and modern pilgrimages to eremi and grottoes where pre-Christian traditions survive under Christian forms, unlike the broader landscape view in the Galleria del Territorio."
    ),

    "GDA-Sala-3": (
        "Explores the continuity of objects, symbols, and rituals from prehistoric times to the twentieth century, showing how certain forms and motifs survive almost unchanged in Abruzzese popular culture. The focus is on long-term links between ancient amulets, protective devices, and decorative patterns and their later rural and Christian counterparts, rather than on a single period or one specific craft like textiles or ceramics.",
        "Displays everyday tools such as ricottiere, lucerne, fusi, and trapani a volano, alongside magical-ritual objects like ciprea shells, cornetti, arrowhead pendants, and mask-like faces on buildings that echo ancient anti-evil symbols. The room also presents festivals with prehistoric roots, including solstice fires, carnival figures, agrarian fertility rites like the ballo della pupa, and Easter pastries shaped as hearts, dolls, and horses, complementing but not duplicating the detailed textile work of GDA-Sala-11 or marriage jewelry of GDA-Sala-12."
    ),

    "GDA-Sala-4": (
        "Room dedicated to the clothing, equipment, and everyday world of Abruzzo shepherds, showing how they dressed, defended themselves, and crafted their own tools in a self-sufficient economy. It emphasizes sheepskin jackets, leather leggings, chiochie sandals, and the use of staffs, slings, umbrellas, and bags designed for a hard outdoor life on the move with flocks, rather than the architecture of stone huts or the legal aspects of transhumance.",
        "Includes objects that highlight the shepherd as artisan and warrior, such as the mazza chiodata for defense, the mazzafionne sling inherited from ancient Italic slingers, carved wooden furniture and gifts, musical instruments like zampogna and ciaramella, and the crucial role of the Pastore Abruzzese-Maremmano dog with its spiked collar. The room underlines how pastoral work, pauses during grazing, and isolation produced a strong craft tradition and a heroic, story-telling culture among shepherds, distinct from the hut reconstructions of GDA-Sala-5 and GDA-Sala-6."
    ),

    "GDA-Sala-5": (
        "Presents the stone pastoral huts known as tholos and the broader world of transhumant shepherding in Abruzzo, focusing on how seasonal movements shaped settlements and economic life. It explains why shepherds needed dry-stone shelters in high mountains, how these were built without mortar, and how the abundance of stone and the practice of monticazione favored this architecture, as opposed to the domestic rural houses shown in GDA-Sala-10.",
        "Displays models and photographs of tholos villages on the Maiella and Gran Sasso, documents about the ancient and early-modern sheep economy, and maps of tratturi used for long-distance migrations between Abruzzo and Puglia. It also includes images of key tasks such as washing, branding, and shearing sheep, contracts and travel permits issued by the Bourbon state, and counting devices and registers used to manage large flocks and pay shepherds, complementing but not repeating the interior life-size hut of GDA-Sala-6."
    ),

    "GDA-Sala-6": (
        "Contains a life-size reconstruction of a stone tholos hut and shows how a shepherd actually lived inside, with minimal furniture and tools arranged for survival in harsh mountain conditions. The architecture demonstrates how corbelled stones form a self-supporting dome without wood or mortar, solving the problem of roofing in a landscape where timber is scarce but stone is abundant, going into more physical detail than the models and photos of GDA-Sala-5.",
        "The room also presents the arciclocco storage pole with hanging cauldrons and friscelle for cheese-making, highlighting the production of pecorino and other dairy foods as essential to pastoral life. Panels explain how centuries of such existence forged key Abruzzese traits such as frugality, toughness, solidarity, and low criminality, and describe the stazzo, a mobile fence enclosure used to protect the flock at night and moved with the shepherd during transhumance or monticazione; questions about shepherd character and identity rather than routes or contracts belong here."
    ),

    "GDA-Galleria-Armi-Guerrieri": (
        "Large gallery tracing the evolution of weapons, armor, and warriors from the Copper Age through the Bronze and Iron Ages to the Roman period and the early Middle Ages, with a special focus on Abruzzo finds. It connects archaeological objects with the broader history of warfare, showing how new metals, tactics, and social structures changed the way conflicts were fought, rather than focusing on peaceful rural life or domestic crafts.",
        "Displays blades, spearheads, helmets, shields, and circular bronze cuirasses typical of Italic warriors, along with the reconstructed grave 302 from the necropolis of Fossa and the full panoply of a Longobard fighter. The exhibition explains hoplite tactics, the rise of organized city-state armies, the professionalization of the Roman legion, and includes didactic areas where students can wear replica helmets, armor, and belts to experience ancient military equipment, complementing the more general prehistory of GDA-Sala-1."
    ),

    "GDA-Sala-7": (
        "Room devoted to traditional cereal agriculture and the annual grain cycle in HISTORICAL rural Abruzzo (mainly early modern to 20th century), from plowing and sowing to harvesting, threshing, winnowing, and storage. It shows how techniques and tools for working wheat remained stable in the world of contadini and sharecroppers, but it is NOT about Paleolithic or Neolithic farmers or prehistoric food – those belong to GDA-Sala-1.",
        "The central model and displays present animal-drawn aratri and erpici, hand sowing a spaglio, sickles and protective finger thimbles, correggiati for threshing, and wooden forks and shovels used for winnowing grain in the wind, along with measures for cereals and tools to protect and store harvests. Use this room for questions about the work, tools and environment of historic peasants and 18th–20th century agricultural life (harvest methods, scarecrows, field huts, measures, rural cereal economy), NOT for questions about prehistoric or Neolithic diets or the very first farmers."
    ),

    "GDA-Sala-8": (
        "Thematic room split into two sectors: traditional transport systems and olive cultivation with oil production, both central to Abruzzo rural life beyond cereal farming. It explains how goods, water, firewood, and crops were moved by human carriers, pack animals, sleds, and carts over steep and often poorly maintained roads, as well as how olives were harvested and processed, without dealing in depth with wine or pork production (which belong to GDA-Sala-9).",
        "Displays photographs and reconstructions of women carrying loads on the head with a spare cloth ring, mules equipped with decorated basti, wooden sleds for steep slopes, and parts of painted wagons. In the agricultural section it illustrates the olive harvest in November, cleaning and bagging of fruit, and the functioning of the frantoio with stone mill, press, fiscoli, and hearth, describing oil as cooking fat, lamp fuel, and medicinal remedy, together with tools and practices for mowing, drying, and storing hay in barns or outdoor haystacks."
    ),

    "GDA-Sala-9": (
        "Room that continues the story of rural food production by focusing on viticulture, winemaking, and pig husbandry as pillars of domestic self-sufficiency in Abruzzo. It presents the historical development from pre-Roman wine culture through predominantly home-consumption vineyards to modern DOC labels like Montepulciano d'Abruzzo, Cerasuolo, Trebbiano, and Controguerra, clearly separate from olive-oil production (GDA-Sala-8) and grain agriculture (GDA-Sala-7).",
        "The displays show grape harvest in baskets, pressing in stone or wooden vats called mese, traditional and mechanical presses, fermentation into novello wine, and storage in oak barrels for aging. The second half of the room is dedicated to the slaughter and complete use of the pig, including hanging carcasses, sausage machines, preserved cuts, and a variety of conservation methods such as salting, smoking, oil and vinegar packing, along with mortars and containers used for sauces, spices, jams, lard, blood puddings, and preserved tomatoes."
    ),

    "GDA-Sala-10": (
        "Room focused on rural housing and domestic life, analyzing how Abruzzo homes were built and organized in mountain, hill, and coastal zones, and how architecture reflected economic conditions. It contrasts stone houses often embedded in rock, earth-and-straw dwellings, and brick or tuff constructions, and then moves inside to examine the division of work spaces and living quarters, rather than pastoral huts (GDA-Sala-5 and GDA-Sala-6) or prisons (GDA-Ceti-Urbani_Risorgimento).",
        "Exhibits the lower level spaces such as stables, barns, cellars, and storerooms, and the upper domestic areas centered on the kitchen with fireplace, bread oven, and simple furniture like tables, benches, and the madia for flour and bread. The room highlights children's toys fashioned from cheap or recycled materials, describes the heavy domestic workload of women including water-carrying and washing, and shows beds with straw or wool mattresses, chests for trousseaux, cradles, and terracotta or metal oil lamps that illuminated the house at night."
    ),

    "GDA-Sala-11": (
        "Explores the complete production cycle of textile fibers, especially linen and wool, in a context where spinning and weaving were mainly domestic and female tasks aimed at family self-sufficiency. It follows the process from field to finished fabric, explaining sowing, harvesting, seed extraction, retting in water, drying, breaking, and combing the fibers before spinning, focusing on techniques rather than on the social rituals of clothing at weddings (GDA-Sala-12).",
        "Displays tools such as the wooden trocche for breaking stems, the ràscele combs, distaff and spindle, and later the small spinning wheel known as felarelle, as well as looms commissioned from carpenters and used in many households. The room also presents simple and patterned linens for sheets, towels, and tablecloths, discusses the flourishing wool craft tradition of mountain centers like Sulmona, Scanno, and Taranta Peligna, and shows blankets and carpets called tarante decorated with geometric and symbolic motifs such as flower vases, trees of life, animals, and stylized human figures."
    ),

    "GDA-Sala-12": (
        "Room dedicated to the life-cycle moment of marriage and the way clothing, dowries, and jewelry expressed social status, gender roles, and values in traditional Abruzzo society. It follows the sequence from courtship and family agreements through the formal promise, exchange of gifts, and preparation of the bride's corredo to the celebration and transfer to the groom's house, focusing on costumes and ornaments rather than on everyday textile production techniques (GDA-Sala-11).",
        "Exhibits wedding and festive costumes, including dark or pastel dresses, traditional rural outfits, head coverings signaling whether a woman is single, married, or widowed, and the rare red eighteenth-century gown from Scanno. The room also focuses on jewelry and amulets such as filigree cannatora necklaces, presentosa pendants with hearts, sciacquajje earrings, children's protective charms and noisy contromalucchie, and a variety of silver buttons, buckles, pins, and multifunctional accessories that combine beauty, symbolism, and magical protection."
    ),

    "GDA-Sala-13": (
        "Room entirely devoted to Abruzzese maiolica and related ceramics, tracing their development from medieval times through the Renaissance, Baroque, and modern periods, and highlighting Abruzzo as one of the most important production areas in Western Europe. It explains what maiolica is, with its tin-glazed surface and painted decoration, and contrasts it with ingobbiata, invetriata, and graffita wares popular in the fifteenth century, focusing on ceramic art rather than on metalwork, textiles, or jewelry.",
        "Shows luxury tableware, pharmacy jars, flower vases, shaving basins, devotional plaques, architectural elements like the famous San Donato ceiling from Castelli, and floor tiles and kitchen linings. It also presents the specialized ceramic towns such as Castelli, Anversa degli Abruzzi, Tagliacozzo, and Torre de' Passeri, then follows the eighteenth- and nineteenth-century decline toward cheaper, popular wares like simple bowls, pitchers, and scaldamani, as markets for aristocratic and export ceramics contracted."
    ),

    "GDA-Galleria-Territorio": (
        "Panoramic gallery presenting Abruzzo as a 'museum in the open air', emphasizing its variety of landscapes from Adriatic coast to high Apennine peaks, the high proportion of land in national and regional parks, and its exceptional biodiversity. It shows how centuries of relative isolation preserved traditional environments, farming systems, and settlement patterns more than in many other Italian regions, giving a territorial overview instead of focusing on one craft or social group.",
        "Panels and images highlight historic villages and town centers of ancient origin, the great artisanal traditions in ceramics, metalwork, textiles, and wood, and the special spiritual landscape of eremi and monasteries that shaped Abruzzese mentality. The gallery also explores the dense network of castles and fortified sites, explains how conserved original contexts reduce the need for museums, and reflects on the advantages and challenges of treating the whole region as a territory-museum where heritage remains embedded in its authentic surroundings."
    ),

    "GDA-Ceti-Urbani_Risorgimento": (
        "Section devoted to the Borbonic prison in Pescara and the rise of modern urban bourgeois society in Abruzzo during the eighteenth and nineteenth centuries, especially in the context of the Risorgimento. It recounts how many young Abruzzese revolutionaries and members of the educated middle classes were chained and incarcerated here between 1850 and 1860 for political crimes linked to demands for Italian unity, constitution, and civil rights, unlike the rural or prehistoric focus of most other rooms.",
        "Provides quantitative data on the bagno penale, the age and social background of the one hundred political prisoners, and the contrast between urban bourgeois modernizers and largely rural masses resistant to change. The displays also examine bourgeois cultural spaces such as palatial houses, salotti, cafés, and theaters where public opinion, secret societies like the Carboneria, and the idea of press freedom were formed, illustrating a shift from feudal structures to a mass society shaped by industrialization and global communication."
    ),
    "GDA-Info-Museo": (
        "Sala dedicata alle informazioni pratiche sul Museo delle Genti d'Abruzzo e sul Museo Basilio Cascella: orari di apertura, prezzi dei biglietti, riduzioni e ingressi gratuiti, come arrivare, contatti, orari della biblioteca e del bookshop, servizi al pubblico e note su mostre ed eventi.",
        "Room dedicated to practical information about the Genti d'Abruzzo Museum and the Basilio Cascella Museum: opening hours, ticket prices, discounts and free admission, how to get there, contact details, library and bookshop hours, visitor services, and notes on exhibitions and events."
    ),

}



# Short descriptions per room for the LLM classifier
ROOM_SHORT_DESC: dict = {}
for rid in ROOM_IDS:
    r = ROOM_DATA[rid]
    custom = CUSTOM_ROOM_DESCRIPTIONS.get(rid)

    if custom is not None:
        # Join tuples/lists of sentences into a single description string
        if isinstance(custom, (tuple, list)):
            desc = " ".join(custom)
        else:
            desc = str(custom)
    else:
        # Fallback: heading + first chunk of room text
        text = (r["text_en"] or r["text_it"]).strip()
        desc = f"{r['heading']}: {text[:240]}"

    ROOM_SHORT_DESC[rid] = desc


# Pre-compute embeddings for room selection (heading + short text)
room_texts_for_emb = []
for rid in ROOM_IDS:
    r = ROOM_DATA[rid]
    base = r["heading"] + "\n" + (r["text_en"] or r["text_it"])
    room_texts_for_emb.append(base[:1000])

if room_texts_for_emb:
    ROOM_EMBS = embed_model.encode(room_texts_for_emb, normalize_embeddings=True)
    ROOM_EMBS = np.asarray(ROOM_EMBS, dtype=np.float32)
else:
    ROOM_EMBS = np.zeros((0, 1), dtype=np.float32)

# -------------------------------------------------------------
# FastAPI models
# -------------------------------------------------------------

app = FastAPI(title="Museum Chatbot (room-level, Qwen)")
app.mount("/app", StaticFiles(directory="web", html=True), name="web")


class HistoryTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class AskReq(BaseModel):
    q: str
    lang: Optional[str] = None          # "it" or "en"; if None we try to guess
    room_id: Optional[str] = None       # optional scoping (QR)
    object_id: Optional[str] = None     # kept for compatibility, unused now
    history: Optional[List[HistoryTurn]] = None  # recent Q/A for pronoun resolution


class Citation(BaseModel):
    url: str
    heading: str
    score: float


class AskResp(BaseModel):
    answer: str
    citations: List[Citation]
    lang: str


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------


def detect_lang(text: str, fallback: str = "it") -> str:
    """Very small heuristic EN/IT detector."""
    t = (text or "").lower()
    # obvious Italian accents
    if re.search(r"[àèéìòóù]", t):
        return "it"
    # common Italian function words
    if any(w in t.split() for w in ("il", "lo", "la", "gli", "le", "per", "non", "che", "come", "quando")):
        return "it"
    # common English function words
    if any(w in t.split() for w in ("the", "and", "what", "who", "when", "where", "why", "how")):
        return "en"
    return fallback


def find_room_id(question: str) -> Optional[str]:
    """Pick the most relevant room for the question using embedding similarity."""
    if ROOM_EMBS.shape[0] == 0:
        return None
    q_emb = embed_model.encode([question], normalize_embeddings=True)[0]
    sims = ROOM_EMBS @ q_emb
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim < ROOM_MIN_SIM:
        return None
    return ROOM_IDS[best_idx]


def build_room_selection_text(question: str, history: Optional[List[HistoryTurn]]) -> str:
    """
    Fuse the latest question with a few previous user prompts so room selection
    can resolve follow-ups like “and what about this painting?”.
    """
    question = (question or "").strip()
    if not history:
        return question

    user_bits: List[str] = []
    for turn in reversed(history):
        if len(user_bits) >= HISTORY_MAX_TURNS:
            break
        if (turn.role or "").lower() != "user":
            continue
        content = (turn.content or "").strip()
        if content:
            user_bits.append(content)

    if not user_bits:
        return question

    user_bits.reverse()
    helper = " ".join(user_bits)
    return f"{question}\n\nPrevious related user questions: {helper}"


def build_history_block(history: Optional[List[HistoryTurn]]) -> str:
    """
    Turn the last N user questions into a small text block like:
    Q: ...
    Used only to resolve pronouns / topic, NOT as factual source.
    """
    if not history:
        return ""
    recent = history[-HISTORY_MAX_TURNS :]
    lines: List[str] = []
    for h in recent:
        role = (h.role or "").lower()
        if role != "user":
            continue
        content = (h.content or "").strip()
        if not content:
            continue
        lines.append(f"Q: {content}")
    block = "\n".join(lines).strip()
    if len(block) > HISTORY_MAX_CHARS:
        block = block[-HISTORY_MAX_CHARS :]
    return block


def answer_logistics(q: str, lang: str) -> str:
    """
    Answer opening hours / tickets / contacts using the museum info text.
    Reuses the same grounded LLM call used for rooms.
    """
    is_en = (lang or "").lower().startswith("en")
    context = MUSEUM_INFO_EN if is_en else MUSEUM_INFO_IT

    # We use the same grounded call as for rooms, but no history
    return call_llm_with_room(
        context=context,
        question=q,
        lang=lang,
        history=None,
    )


def ollama_chat(model: str, system_prompt: str, user_msg: str, tag: str = "LLM", temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        print(f"[{tag}] Calling {model} at {OLLAMA_URL}")
        print(f"[{tag}] system prompt preview: {system_prompt[:120]!r}")
        print(f"[{tag}] user_msg preview: {user_msg[:200]!r}\n")

        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        print(f"[{tag}] HTTP status: {resp.status_code}")
        resp.raise_for_status()

        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()
        print(f"[{tag}] raw reply preview: {content[:200]!r}\n")
        return content
    except Exception as e:
        print(f"[{tag}] ERROR: {e}\n")
        return ""


def get_room_candidates(selector_text: str, top_k: int = 5) -> List[tuple[str, float]]:
    """Return top-k rooms by embedding similarity."""
    if ROOM_EMBS.shape[0] == 0:
        return []
    selector_text = (selector_text or "").strip()
    if not selector_text:
        return []
    q_emb = embed_model.encode([selector_text], normalize_embeddings=True)[0]
    sims = ROOM_EMBS @ q_emb
    order = np.argsort(-sims)[:top_k]
    return [(ROOM_IDS[i], float(sims[i])) for i in order]


def classify_room_with_llm(question: str, lang: str, candidates: List[tuple[str, float]]) -> Optional[str]:
    """
    Use the main 7B model as a classifier over a list of candidate rooms.
    Returns a room_id from candidates, or None on failure.
    """
    if not candidates:
        return None

    is_en = (lang or "").lower().startswith("en")

    # Build candidate list with short descriptions
    lines = []
    for rid, score in candidates:
        desc = ROOM_SHORT_DESC.get(rid, ROOM_DATA[rid]["heading"])
        lines.append(f'- "{rid}": {desc}')
    rooms_block = "\n".join(lines)

    if is_en:
        system_prompt = (
            "You are a classifier for a museum chatbot.\n"
            "Your task is to choose which room best matches the visitor question.\n"
            "You must reply ONLY with a JSON object of the form:\n"
            '{"room_id": "<ID>"}\n'
            "where <ID> is exactly one of the IDs listed in the candidate rooms."
        )
        user_msg = (
            f"Visitor question:\n{question}\n\n"
            f"Candidate rooms:\n{rooms_block}\n\n"
            "Choose the single best room_id and return only the JSON."
        )
    else:
        system_prompt = (
            "Sei un classificatore per una guida museale.\n"
            "Devi scegliere quale sala corrisponde meglio alla domanda del visitatore.\n"
            "Devi rispondere SOLO con un oggetto JSON del tipo:\n"
            '{"room_id": "<ID>"}\n'
            "dove <ID> è esattamente uno degli ID elencati nelle sale candidate."
        )
        user_msg = (
            f"Domanda del visitatore:\n{question}\n\n"
            f"Sale candidate:\n{rooms_block}\n\n"
            "Scegli un solo room_id e restituisci solo il JSON."
        )

    # Use the same 7B model for classification
    txt = ollama_chat(LLM_MODEL, system_prompt, user_msg, tag="ROOM-CLS", temperature=0.0)
    if not txt:
        return None

    # Try JSON parse first
    try:
        obj = json.loads(txt)
        rid = obj.get("room_id")
        if isinstance(rid, str) and any(rid == c[0] for c in candidates):
            return rid
    except Exception:
        pass

    # Fallback: look for a room_id substring
    for rid, _ in candidates:
        if rid in txt:
            return rid

    return None


def select_room_id(question: str, lang: str, history: Optional[List[HistoryTurn]]) -> Optional[str]:
    """
    Decide which room to use.

    We combine the current question with recent user questions so that
    follow-ups like "How many died?" stay in the same room, while still
    letting the classifier choose freely when the topic changes.
    """
    selector_text = build_room_selection_text(question, history)
    selector_text = (selector_text or "").strip()
    if not selector_text:
        return None

    # 1) Try the 7B classifier over all rooms
    candidates = [(rid, 0.0) for rid in ROOM_IDS]
    rid = classify_room_with_llm(selector_text, lang, candidates)
    if rid and rid in ROOM_DATA:
        print(f"[ROOM] LLM classifier chose: {rid}")
        return rid

    print("[ROOM] LLM classifier failed or invalid, trying embeddings.")

    # 2) Fallback: embeddings on the same combined text
    if ROOM_EMBS.shape[0] == 0:
        return None

    q_emb = embed_model.encode([selector_text], normalize_embeddings=True)[0]
    sims = ROOM_EMBS @ q_emb
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_rid = ROOM_IDS[best_idx]
    print(f"[ROOM] embedding best: {best_rid} (sim={best_sim:.3f})")

    if best_sim < ROOM_MIN_SIM:
        print(f"[ROOM] best below threshold {ROOM_MIN_SIM}, abstaining.")
        return None

    return best_rid

    # 3) If we get here, the CURRENT question is ambiguous.
    #    Fallback: default to the room that fits the LAST user question.
    if history:
        last_user_q = None
        for turn in reversed(history):
            if (turn.role or "").lower() != "user":
                continue
            content = (turn.content or "").strip()
            if content:
                last_user_q = content
                break

        if last_user_q:
            print(f"[ROOM] current question ambiguous, using last user question as fallback: {last_user_q!r}")

            # 3a) Try LLM classifier on last question
            rid_prev = classify_room_with_llm(last_user_q, lang, candidates)
            if rid_prev and rid_prev in ROOM_DATA:
                print(f"[ROOM] LLM classifier chose (last question): {rid_prev}")
                return rid_prev

            # 3b) Embedding fallback on last question
            if ROOM_EMBS.shape[0] > 0:
                prev_emb = embed_model.encode([last_user_q], normalize_embeddings=True)[0]
                sims_prev = ROOM_EMBS @ prev_emb
                best_idx_prev = int(np.argmax(sims_prev))
                best_sim_prev = float(sims_prev[best_idx_prev])
                best_rid_prev = ROOM_IDS[best_idx_prev]
                print(f"[ROOM] embedding (last question) best: {best_rid_prev} (sim={best_sim_prev:.3f})")
                if best_sim_prev >= ROOM_MIN_SIM:
                    return best_rid_prev

    print("[ROOM] unable to select a room with confidence, returning None.")
    return None



def build_critic_prompts(
    context: str,
    question: str,
    candidate_answer: str,
    lang: str,
    dont_know: str,
) -> tuple[str, str]:
    """Build system + user prompts for the critic pass."""
    is_en = (lang or "").lower().startswith("en")
    if is_en:
        system_prompt = (
            "You are a strict fact-checker for a museum guide.\n"
            "You receive a room context (official museum text), a visitor question and a candidate answer.\n"
            "Your job is to ensure the final answer is fully supported by the room context."
        )
        user_msg = (
            "Room context:\n"
            f"{context}\n\n"
            "Visitor question:\n"
            f"{question}\n\n"
            "Candidate answer from the guide:\n"
            f"{candidate_answer}\n\n"
            "Tasks:\n"
            f"1. Check whether the candidate answer is fully supported by the room context. "
            f"If any part is not stated or directly implied, consider it unsupported.\n"
            "2. If the candidate is fully supported but could be clearer or slightly more detailed, "
            "rewrite it using only information from the room context.\n"
            f"3. If the candidate includes unsupported information, ignore it and answer again from scratch "
            f"using only the room context. If the context does not contain the answer, say exactly: {dont_know}\n\n"
            "Return only the final answer, not your reasoning."
        )
    else:
        system_prompt = (
            "Sei un rigoroso verificatore di fatti per una guida museale.\n"
            "Ricevi un testo di contesto della sala, una domanda del visitatore e una risposta candidata.\n"
            "Il tuo compito è assicurarti che la risposta finale sia completamente supportata dal contesto."
        )
        user_msg = (
            "Testo di contesto della sala:\n"
            f"{context}\n\n"
            "Domanda del visitatore:\n"
            f"{question}\n\n"
            "Risposta candidata della guida:\n"
            f"{candidate_answer}\n\n"
            "Compiti:\n"
            f"1. Verifica se la risposta candidata è pienamente supportata dal testo di contesto. "
            f"Se qualche parte non è affermata o chiaramente implicata, considerala non supportata.\n"
            "2. Se la risposta è supportata ma può essere più chiara o leggermente più dettagliata, "
            "riscrivila usando solo informazioni presenti nel testo di contesto.\n"
            f"3. Se la risposta contiene informazioni non supportate, ignorale e rispondi da zero usando solo il testo. "
            f"Se il contesto non contiene la risposta, dì esattamente: {dont_know}\n\n"
            "Restituisci solo la risposta finale, non il ragionamento."
        )
    return system_prompt, user_msg


def call_llm_with_room(
    context: str,
    question: str,
    lang: str,
    history: Optional[List[HistoryTurn]] = None,
) -> str:
    """
    Call local Qwen via Ollama with strong grounding + small sliding window.
    Optionally run a second critic pass to self-check the answer.
    """
    lang = (lang or "it").lower()
    is_en = lang.startswith("en")

    if is_en:
        dont_know = "I don't quite know how to answer this question. For more info, please check the website or email a member of staff at museo@gentidabruzzo.it"
        system_prompt = (
            "You are a museum guide at the Genti d'Abruzzo museum.\n"
            "You will receive the full official text for one room (the room context) and a visitor question.\n"
            "Use ONLY the information in the room context to answer the question.\n"
            f"If the room context really does not contain the answer, reply exactly: {dont_know}\n"
            "Always answer in ENGLISH, in at most 3 short sentences."
        )
    else:
        dont_know = "Non lo so, puoi mandare un email a museo@gentidabruzzo.it per informazioni"
        system_prompt = (
            "Sei una guida del Museo delle Genti d'Abruzzo.\n"
            "Riceverai il testo ufficiale di una sala (contesto della sala) e una domanda del visitatore.\n"
            "Usa SOLO le informazioni presenti nel contesto della sala per rispondere.\n"
            f"Se il contesto davvero non contiene la risposta, rispondi esattamente: {dont_know}\n"
            "Rispondi sempre in ITALIANO, in massimo 3 frasi brevi."
        )


    context = (context or "").strip()
    if len(context) > MAX_CTX_CHARS:
        context = context[:MAX_CTX_CHARS]

    history_block = build_history_block(history)

    user_msg_parts = [
        "Room context:",
        context,
        "",
    ]
    if history_block:
        user_msg_parts.extend(
            [
                "Recent visitor questions (for pronouns/topic only; do NOT contradict or extend the room context):",
                history_block,
                "",
            ]
        )

    if is_en:
        last_line = (
        "Answer in ENGLISH ONLY in 2–3 short sentences, quoting the key facts "
        "(names, numbers, dates) from the room context. Do NOT reply in Italian. "
        "Do not add any facts."
    )
    else:
        last_line = (
        "Rispondi SOLO IN ITALIANO in 2–3 frasi brevi, riportando i fatti principali "
        "(nomi, numeri, date) dal testo di contesto. Non rispondere in inglese. "
        "Non aggiungere fatti."
    )


    user_msg_parts.extend(
        [
            "New question:",
            question,
            "",
            last_line,
        ]
    )
    user_msg = "\n".join(user_msg_parts)

    # First pass: candidate answer
    answer = ollama_chat(LLM_MODEL, system_prompt, user_msg, tag="LLM", temperature=0.0)
    if not answer:
        answer = dont_know

    # Optional critic pass
    if ENABLE_CRITIC:
        critic_system, critic_user = build_critic_prompts(context, question, answer, lang, dont_know)
        critic_answer = ollama_chat(CRITIC_MODEL, critic_system, critic_user, tag="CRITIC", temperature=0.0)
        if critic_answer:
            answer = critic_answer

    return answer or dont_know


# -------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------


@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    q = (req.q or "").strip()
    if not q:
        lang = (req.lang or "it").lower()
        msg = "Domanda vuota." if not lang.startswith("en") else "Empty question."
        return AskResp(answer=msg, citations=[], lang=lang)

    # language: detect from text first
    auto_lang = detect_lang(q)          # "it" or "en"
    lang = auto_lang

    # If the client explicitly passes a lang AND it matches the detection, keep it.
    # If it disagrees (IT UI but EN text), trust the text.
    if req.lang:
        req_lang = req.lang.lower()
        if req_lang.startswith(auto_lang):
            lang = req_lang  # they agree, fine
        else:
            # Mismatch: log it but prefer the language inferred from the question text
            print(f"[LANG] UI lang={req.lang} but text looks like {auto_lang}; using {auto_lang}.")
            lang = auto_lang

    is_en = lang.startswith("en")

    # --------------------------------------------------
    # Room selection, with special handling for logistics
    # --------------------------------------------------
    if OFFTOPIC_RE.search(q):
        # Force the synthetic "museum info" room and skip classifier
        room_id = INFO_ROOM_ID
        print(f"[ASK] logistics question detected, forcing room_id={room_id}")
    else:
        if req.room_id:
            room_id = req.room_id
        else:
            room_id = select_room_id(q, lang, req.history)




    if not room_id or room_id not in ROOM_DATA:
        msg = (
            "Non lo so. Non riesco a capire a quale sala si riferisce la domanda."
            if not is_en
            else "I don't know. I couldn't determine which room this question refers to."
        )
        return AskResp(answer=msg, citations=[], lang=lang)

    # --------------------------------------------------
    # Build context from the chosen room
    # --------------------------------------------------
    room = ROOM_DATA[room_id]

    # For English, prefer curated English text; otherwise use Italian text
    if is_en and room.get("text_en"):
        context = room["text_en"]
    else:
        context = room["text_it"]

    # DEBUG: show which room and how much context we are sending
    print(f"[ASK] lang={lang} room_id={room_id} heading={room['heading']!r}")
    print(f"[ASK] context length = {len(context)} chars")
    print(f"[ASK] context preview = {context[:200]!r}\n")

    # --------------------------------------------------
    # Call local LLM with room context + (optional) history
    # --------------------------------------------------
    answer = call_llm_with_room(
        context=context,
        question=q,
        lang=lang,
        # For the museum info room we ignore chat history
        history=None if room_id == INFO_ROOM_ID else req.history,
    )

    # If the model says it doesn't know, always point to staff / website / contacts
    dont_know_en = "I don't know, please check the website for more information"
    dont_know_it = "Non lo so sulla base del testo fornito, per queste informazioni chiedi al personale"

    if is_en and dont_know_en in answer:
        answer = (
            f"{dont_know_en} "
            "For this information, please ask a member of staff or contact the museum at "
            "+39 085 451 0026 or museo@gentidabruzzo.it, or check the official website."
        )

    if (not is_en) and dont_know_it in answer:
        answer = (
            f"{dont_know_it} "
            "Per queste informazioni chiedi al personale oppure contatta il museo al "
            "+39 085 451 0026 o via email a museo@gentidabruzzo.it, "
            "oppure consulta il sito ufficiale."
        )


    citations: List[Citation] = []
    if room.get("url"):
        citations.append(
            Citation(
                url=room["url"],
                heading=room["heading"],
                score=1.0,
            )
        )

    return AskResp(answer=answer, citations=citations, lang=lang)



@app.get("/healthz")
def healthz():
    return {"ok": True, "rooms": len(ROOM_IDS)}
