import streamlit as st
import time
import getpass
from neo4j import GraphDatabase
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from neo4j import GraphDatabase
# Removed: from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List, Dict, Any
import re
import requests
from datetime import datetime
from cachetools import cached, TTLCache
import graphviz
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from operator import add
import unicodedata
from langgraph.checkpoint.memory import MemorySaver
import random
import os
import uuid

# Load secrets directly from Streamlit's secrets management
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    URI = st.secrets["NEO4J_URI"]
    USERNAME = st.secrets["NEO4J_USER"]
    PASSWORD = st.secrets["NEO4J_PASSWORD"]
    NEO4J_DATABASE = st.secrets.get("NEO4J_DATABASE", "neo4j")
    st.success("API keys and Neo4j credentials loaded successfully from Streamlit secrets.")
except KeyError as e:
    st.error(f"Missing Streamlit secret: {e}. Please configure this in your Streamlit Cloud dashboard.")
    st.stop()


def sanitize_markdown(text):
    if not isinstance(text, str):
        st.warning(f"sanitize_markdown received non-string input: {type(text)} - {text}")
        return str(text)
    if not text:
        return ""

    # HTML özel karakterleri
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Kaçırılması gereken Markdown karakterleri
    markdown_chars = ['\\', '*', '_', '~', '`', '#', '[', ']', '(', ')', '{', '}', '!', '^']
    for char in markdown_chars:
        text = text.replace(char, f"\\{char}")

    return text


def safe_markdown(text):
    try:
        # Basitçe regex uyumluluğunu kontrol et
        re.compile(text)
        return text
    except re.error:
        # Regex hatası varsa, tüm metni saf metin gibi göster
        return f"<pre>{text}</pre>"

# --- Neo4j Bağlantı Sınıfı ---
class Neo4jConnector:
    def __init__(self):
        # These should now read from st.secrets directly
        self.uri = st.secrets["NEO4J_URI"] # Corrected
        self.user = st.secrets["NEO4J_USER"] # Corrected
        self.password = st.secrets["NEO4J_PASSWORD"] # Corrected
        self.database = st.secrets.get("NEO4J_DATABASE", "neo4j") # Corrected
        self.driver = None

    def connect(self):
        """Establishes a connection to Neo4j."""
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                with self.driver.session(database=self.database) as session:
                    session.run("RETURN 1")
                st.info("Neo4j bağlantısı başarılı.")
            except Exception as exc:
                st.error(f"Neo4j bağlantı hatası: {exc}")
                raise ConnectionError(f"Neo4j bağlantı hatası: {exc}") from exc

    def close(self):
        """Closes the Neo4j driver if it's open."""
        if self.driver:
            self.driver.close()
            self.driver = None
            st.info("Neo4j bağlantısı kapatıldı.")


    def get_meyhaneler(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Fetches 'Meyhane' nodes from Neo4j in the format expected by your LangGraph app.
        """
        self.connect()
        query = """
        MATCH (m:Meyhane)
        RETURN
            m.name                  AS name,
            m.google_adres          AS address,
            m.google_ortalama_puan AS rating,
            m.google_toplam_yorum   AS review_count,
            m.maps_linki            AS map_link,
            m.google_telefon        AS phone,
            m.fiyat_seviyesi_simge AS price_level,
            elementId(m)            AS neo4j_element_id
        ORDER BY m.google_ortalama_puan DESC
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, limit=limit)
                records = [self._clean_record(record) for record in result]
                st.info(f"Neo4j'den {len(records)} mekan çekildi.")
                return records
        except Exception as exc:
            st.error(f"Neo4j sorgu hatası (get_meyhaneler): {exc}")
            print(f"Sorgu hatası (get_meyhaneler): {exc}")
            return []

    @staticmethod
    def _clean_record(record) -> Dict[str, Any]:
        """
        Cleans Neo4j record by replacing None values with defaults
        and ensuring correct types for numeric fields.
        """
        name = record.get("name") or "Bilinmiyor"
        address = record.get("address") or "Adres yok"
        rating_value = record.get("rating")
        rating = float(rating_value) if rating_value is not None else 0.0
        review_count_value = record.get("review_count")
        review_count = int(review_count_value) if review_count_value is not None else 0
        map_link = record.get("map_link") or ""
        phone = record.get("phone") or ""
        price_level = record.get("price_level") or ""
        neo4j_element_id = record["neo4j_element_id"]

        return {
            "name": name,
            "address": address,
            "rating": rating,
            "review_count": review_count,
            "map_link": map_link,
            "phone": phone,
            "price_level": price_level,
            "neo4j_element_id": neo4j_element_id,
        }

# --- LangGraph State Tanımlaması ---
def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Combines two lists of BaseMessage, used for state annotation."""
    return left + right

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    last_recommended_place: Optional[str]
    next_node: Optional[str]
    location_query: Optional[str]

def process_documents(docs: List[Any]) -> List[Document]:
    processed = []
    for doc in docs:
        if isinstance(doc, dict):
            metadata = {
                "Mekan Adı": doc.get("name", "Bilinmeyen Mekan"),
                "Adres": doc.get("address", "Bilinmeyen Adres"),
                "Google Puanı": str(doc.get("rating", 0.0)),
                "Google Yorum Sayısı": str(doc.get("review_count", 0)),
                "Maps Linki": doc.get("map_link", "Harita linki yok"),
                "Telefon": doc.get("phone", "Yok"),
                "Fiyat Seviyesi": str(doc.get("price_level", "Yok"))
            }
            main_content = (
                f"Mekan Adı: {metadata['Mekan Adı']}, "
                f"Adres: {metadata['Adres']}, "
                f"Google Puanı: {metadata['Google Puanı']}, "
                f"Google Yorum Sayısı: {metadata['Google Yorum Sayısı']}, "
                f"Fiyat Seviyesi: {metadata['Fiyat Seviyesi']}"
            )
            processed.append(Document(
                page_content=main_content,
                metadata=metadata
            ))
    st.info(f"LangChain için {len(processed)} doküman işlendi.")
    return processed

# --- Neo4j Verilerini Yükle ve Vektör Deposunu Oluştur (Sadece Bir Kez Çalıştır) ---
@st.cache_resource
def initialize_retriever():
    st.info("Retriever başlatılıyor...")
    meyhaneler_listesi = []
    try:
        conn = Neo4jConnector()
        meyhaneler_listesi = conn.get_meyhaneler(limit=10000)
        conn.close()
        if not meyhaneler_listesi:
            st.warning("Uyarı: Neo4j'den hiç mekan verisi çekilemedi. Retrieval boş sonuç dönebilir. Dummy veri kullanılıyor.")
            meyhaneler_listesi = [
                {"name": "Dummy Meyhane A", "address": "Dummy Adres A", "rating": 4.0, "review_count": 100, "map_link": "http://dummy.map.a", "phone": "000", "price_level": 2, "neo4j_element_id": "dummy-a"},
                {"name": "Dummy Meyhane B", "address": "Dummy Adres B", "rating": 4.5, "review_count": 250, "map_link": "http://dummy.map.b", "phone": "000", "price_level": 3, "neo4j_element_id": "dummy-b"},
                {"name": "Dummy Meyhane C", "address": "Dummy Adres C", "rating": 3.8, "review_count": 50, "map_link": "http://dummy.map.c", "phone": "000", "price_level": 1, "neo4j_element_id": "dummy-c"},
            ]
            st.info(f"Dummy veri kullanılıyor: {len(meyhaneler_listesi)} mekan.")
        else:
            st.info(f"Neo4j'den gerçek veri kullanılıyor: {len(meyhaneler_listesi)} mekan.")
    except Exception as e:
        st.error(f"Neo4j'den veri çekerken hata oluştu: {e}. Lütfen Neo4j sunucunuzun çalıştığından ve kimlik bilgilerinin doğru olduğundan emin olun. Dummy veri kullanılıyor.")
        meyhaneler_listesi = [
            {"name": "Dummy Meyhane A", "address": "Dummy Adres A", "rating": 4.0, "review_count": 100, "map_link": "http://dummy.map.a", "phone": "000", "price_level": 2, "neo4j_element_id": "dummy-a"},
            {"name": "Dummy Meyhane B", "address": "Dummy Adres B", "rating": 4.5, "review_count": 250, "map_link": "http://dummy.map.b", "phone": "000", "price_level": 3, "neo4j_element_id": "dummy-b"},
            {"name": "Dummy Meyhane C", "address": "Dummy Adres C", "rating": 3.8, "review_count": 50, "map_link": "http://dummy.map.c", "phone": "000", "price_level": 1, "neo4j_element_id": "dummy-c"},
        ]
        st.info(f"Dummy veri kullanılıyor: {len(meyhaneler_listesi)} mekan.")

    processed_docs = process_documents(meyhaneler_listesi)
    try:
        vectorstore = InMemoryVectorStore.from_documents(
            documents=processed_docs,
            embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # Pass API key explicitly
        )
        st.success("Vektör deposu başarıyla oluşturuldu.")
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Vektör deposu oluşturulurken hata oluştu: {e}. OpenAI API anahtarınızı kontrol edin.")
        # Return a dummy retriever to prevent crash
        class DummyRetriever:
            def invoke(self, query, k):
                st.warning("Dummy retriever kullanılıyor. Gerçek arama yapılamıyor.")
                return [Document(page_content="Dummy Mekan", metadata={"Mekan Adı": "Dummy Mekan", "Adres": "Bilinmiyor", "Google Puanı": "0.0", "Google Yorum Sayısı": "0", "Maps Linki": "", "Telefon": "", "Fiyat Seviyesi": ""})]
        return DummyRetriever()

retriever = initialize_retriever()

# --- LLM ve Prompt Tanımlamaları ---
SYSTEM_PROMPT = """Sen İstanbul'da romantik mekan, meyhane, restoran ve kafe önerisi yapabilen bir AI asistanısın.
Kullanıcıya Google haritalar bilgileriyle desteklenmiş, hava durumuyla uyumlu önerilerde bulunabilirsin.
Gelen sorulara doğal, nazik ve samimi bir dille cevap ver ve tüm cevapların Türkçe olsun.

Aşağıdaki gibi konuşmalar seni yönlendirmelidir:
- 'Beşiktaş’ta romantik bir mekan var mı?' → Mekan araması yap
- 'Yarın Beşiktaş'ta hava nasıl olacak?' → Hava durumu kontrol et
- 'Bir ilginç bilgi ver' → Eğlenceli bir bilgi paylaş
- 'Merhaba', 'Selam' → Karşılama mesajı gönder

Kullandığın veritabanında yer alan mekanlar sadece İstanbul sınırları içindedir.
Eğer kullanıcı başka şehirde mekan istiyorsa, bunu açıkça belirtmelisin."""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Sohbet geçmişini kısa ve öz şekilde özetle. Sadece önemli bilgileri korla."),
    MessagesPlaceholder(variable_name="messages"),
])

# --- Yardımcı Fonksiyonlar ---
@cached(TTLCache(maxsize=100, ttl=3600))
def get_fun_fact() -> str:
    st.info("İlginç bilgi alınıyor...")
    try:
        response = requests.get("https://uselessfacts.jsph.pl/api/v2/facts/random?language=tr", timeout=5)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        fact = response.json().get("text", "İlginç bir bilgi bulunamadı.")
        st.success(f"İlginç bilgi alındı: {fact[:50]}...")
        return fact
    except requests.exceptions.Timeout:
        st.error("İlginç bilgi servisi zaman aşımına uğradı.")
        return "İlginç bilgi servisi şu an çok yavaş veya çalışmıyor."
    except requests.exceptions.RequestException as e:
        st.error(f"İlginç bilgi servisi hatası: {e}")
        return f"İlginç bilgi servisi şu an çalışmıyor. Hata: {e}"
    except Exception as e:
        st.error(f"İlginç bilgi alınırken beklenmedik hata: {e}")
        return f"İlginç bilgi alınırken beklenmedik bir hata oluştu: {e}"

def clean_location_query(query: str) -> str:
    normalized_query = unicodedata.normalize('NFKD', query.lower()).encode('ascii', 'ignore').decode('utf-8')
    st.info(f"Konum sorgusu temizleniyor: '{query}' -> '{normalized_query}'")

    istanbul_locations = [
        r'etiler', r'levent', r'maslak', r'nisantasi', r'nisantaşi',
        r'bebek', r'arnavutkoy', r'arnavutköy', r'ortakoy', r'ortaköy', r'cihangir',
        r'taksim', r'karakoy', r'karaköy', r'galata', r'fatih',
        r'sultanahmet', r'eminonu', r'eminönü', r'kadikoy', r'kadıköy', r'moda',
        r'bagdat caddesi', r'bağdat caddesi', r'suadiye', r'bostanci', r'bostancı',
        r'maltepe', r'kartal', r'pendik', r'uskudar', r'üsküdar',
        r'camlica', r'çamlıca', r'beykoz', r'atasehir', r'ataşehir', r'cekmekoy', r'çekmeköy',
        r'sariyer', r'sarıyer', r'istinye', r'tarabya', r'yenikoy', r'yeniköy',
        r'bahcekoy', r'bahçeköy', r'buyukdere', r'büyükdere', r'zumrutevler', r'zümrutevler',
        r'florya', r'yesilkoy', r'yeşilköy', r'yesilyurt', 'yeşilyurt', r'bakirkoy', r'bakırköy',
        r'atakoy', r'ataköy', r'zeytinburnu', r'gungoren', r'güngören', r'esenler',
        r'bayrampasa', r'bayrampaşa', r'gaziosmanpasa', r'gaziosmanpaşa', r'eyup', r'eyüp', r'kagithane', r'kağıthane',
        r'sisli', r'şişli', r'besiktas', r'beşiktaş', r'avcilar', r'avcılar', r'beylikduzu', 'beylikdüzü',
        r'esenyurt', r'buyukcekmece', r'büyükçekmece', r'silivri', r'catalca', r'çatalca',
        r'sile', r'şile', r'agva', r'ağva', r'adalar', r'basaksehir', 'başakşehir',
        r'bahcelievler', r'bahçelievler', r'kucukcekmece', r'küçükçekmece', r'cankurtaran'
    ]

    for loc_regex in istanbul_locations:
        match = re.search(r'\b' + loc_regex + r'\b', normalized_query)
        if match:
            st.info(f"Konum bulundu (İstanbul): {match.group(0)}")
            return match.group(0)

    general_cities = [
        r'istanbul', r'ankara', r'izmir', r'adana',
        r'bursa', r'antalya', r'konya', r'kayseri',
        r'gaziantep', r'samsun', r'eskisehir', r'eskişehir', r'duzce', r'düzce'
    ]

    for city_regex in general_cities:
        match = re.search(r'\b' + city_regex + r'\b', normalized_query)
        if match:
            st.info(f"Konum bulundu (Genel Şehir): {match.group(0)}")
            return match.group(0)

    st.info("Konum bulunamadı, 'istanbul' varsayılan olarak ayarlandı.")
    return "istanbul"

weather_cache = TTLCache(maxsize=100, ttl=300)

@cached(weather_cache)
def get_openweather_forecast(location: str) -> Dict:
    st.info(f"Hava durumu tahmini alınıyor: {location}")
    api_key = st.secrets.get("OPENWEATHER_API_KEY") # Corrected
    if not api_key:
        st.error("OpenWeather API anahtarı bulunamadı.")
        return {"error": "API anahtarı bulunamadı."}
    try:
        geo_response = requests.get(
            f"http://api.openweathermap.org/geo/1.0/direct?q={location},TR&limit=1&appid={api_key}",
            timeout=10,
        )
        geo_response.raise_for_status()
        geo = geo_response.json()
        if not geo:
            st.warning(f"'{location}' konumu için coğrafi veri bulunamadı.")
            return {"error": f"'{location}' konumu bulunamadı."}
        lat, lon = geo[0]["lat"], geo[0]["lon"]
        st.info(f"Coğrafi koordinatlar: Lat={lat}, Lon={lon}")

        weather_response = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=tr",
            timeout=10,
        )
        weather_response.raise_for_status()
        weather = weather_response.json()
        st.success(f"{location} için hava durumu verisi başarıyla alındı.")
        return weather
    except requests.exceptions.RequestException as e:
        st.error(f"OpenWeather API hatası: {e}")
        return {"error": f"API hatası: {e}"}
    except Exception as e:
        st.error(f"Hava durumu verisi alınırken beklenmedik hata: {e}")
        return {"error": f"Beklenmedik bir hata oluştu: {e}"}

def format_weather_response(location: str, data: Dict) -> str:
    st.info(f"Hava durumu yanıtı formatlanıyor for {location}...")
    if "error" in data:
        st.error(f"Hava durumu formatlama hatası: {data['error']}")
        return f"❌ {data['error']}"
    try:
        lines = [f"🌤️ **{location.capitalize()} Hava Durumu Tahmini:**"]
        if "list" not in data or not data["list"]:
            st.warning(f"{location.capitalize()} için hava durumu listesi boş.")
            return f"❌ {location.capitalize()} için hava durumu tahmini bulunamadı."

        for item in data.get("list", [])[:8]: # Sonraki 24 saat için (3 saatte bir)
            dt = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")
            lines.append(
                f"• {dt:%d/%m %H:%M}: "
                f"{item['weather'][0]['description'].capitalize()}, "
                f"Sıcaklık: {item['main']['temp']}°C, "
                f"Hissedilen: {item['main']['feels_like']}°C, "
                f"Nem: {item['main']['humidity']}%, "
                f"Rüzgar: {item['wind']['speed']} m/s"
            )
        formatted_string = "\n".join(lines)
        st.success("Hava durumu yanıtı başarıyla formatlandı.")
        return formatted_string
    except Exception as e:
        st.error(f"Hava durumu verisi işlenirken hata oluştu: {e}")
        return f"❌ Hava durumu verisi işlenirken hata oluştu: {e}"

# --- LangGraph Nodes ---
def check_weather_node(state: AgentState) -> AgentState:
    st.info("`check_weather_node` çalışıyor.")
    last_msg = state["messages"][-1]
    query = last_msg.content

    location = state.get("location_query") or clean_location_query(query)
    state["location_query"] = location
    st.info(f"Hava durumu sorgulanacak konum: {location}")

    formatted = format_weather_response(location, get_openweather_forecast(location))

    sanitized_formatted = sanitize_markdown(formatted)
    state["messages"].append(AIMessage(content=sanitized_formatted))
    st.success("Hava durumu yanıtı AIMessage olarak eklendi.")
    return state

def add_system_message(state: AgentState) -> AgentState:
    st.info("`add_system_message` çalışıyor.")
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
        st.info("Sistem mesajı eklendi.")
        return {"messages": [system_msg] + state["messages"]}
    else:
        st.info("Sistem mesajı zaten mevcut.")
        return state

def summarize_conversation(state: AgentState) -> AgentState:
    st.info("`summarize_conversation` çalışıyor.")
    messages = state["messages"]

    if len(messages) > 5: # Konuşma belirli bir uzunluğu aşınca özetle
        st.info(f"Sohbet özetleniyor, mesaj sayısı: {len(messages)}")
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY) # Pass API key explicitly
            chain = SUMMARY_PROMPT | llm
            summary = chain.invoke({"messages": messages})
            st.success("Sohbet başarıyla özetlendi.")
            return {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    AIMessage(content=summary.content)
                ]
            }
        except Exception as e:
            st.error(f"Sohbet özetlenirken hata oluştu: {e}")
            # Fallback to current messages if summarization fails
            return state
    else:
        st.info("Sohbet özetlenmedi, mesaj sayısı 5'ten az.")
        return state

def search_meyhaneler_node(state: AgentState) -> AgentState:
    st.info("`search_meyhaneler_node` çalışıyor.")
    last_msg = state["messages"][-1]
    query = last_msg.content
    st.info(f"Arama sorgusu: {query}")

    location = clean_location_query(query)
    state["location_query"] = location
    st.info(f"Arama konumu belirlendi: {location}")

    try:
        # Arama sorgusunu iyileştir
        retrieval_query = f"{location} {query}" if location and location not in query.lower() else query
        st.info(f"Retrieval sorgusu: {retrieval_query}")
        raw_results = retriever.invoke(retrieval_query, k=10) # Daha fazla sonuç alıp filtreleyelim
        st.info(f"Retriever'dan {len(raw_results)} ham sonuç alındı.")
        filtered_results = []

        normalized_location = unicodedata.normalize('NFKD', location.lower()).encode('ascii', 'ignore').decode('utf-8')

        # Konuma göre filtreleme
        for doc in raw_results:
            address_lower = unicodedata.normalize('NFKD', doc.metadata.get("Adres", "").lower()).encode('ascii', 'ignore').decode('utf-8')
            name_lower = unicodedata.normalize('NFKD', doc.metadata.get("Mekan Adı", "").lower()).encode('ascii', 'ignore').decode('utf-8')

            # Eğer konum sorguda yer alıyorsa ve mekanın adresi veya adı bu konumu içeriyorsa ekle
            if normalized_location in address_lower or normalized_location in name_lower:
                filtered_results.append(doc)
        st.info(f"Konuma göre filtrelendikten sonra {len(filtered_results)} sonuç kaldı.")

        # Eğer filtrelemeden sonra hala sonuç yoksa veya ilk 3 sonuç yoksa daha geniş arama yap
        if not filtered_results or len(filtered_results) < 3:
            st.info("Filtrelenmiş sonuçlar yetersiz, genel İstanbul araması yapılıyor.")
            # Sadece İstanbul için genel arama
            raw_results_istanbul = retriever.invoke(f"istanbul {query}", k=5)
            # Daha önce filtrelenmiş sonuçları da ekle, mükerrerleri önle
            seen_names = {doc.metadata.get("Mekan Adı") for doc in filtered_results}
            for doc in raw_results_istanbul:
                if doc.metadata.get("Mekan Adı") not in seen_names:
                    filtered_results.append(doc)
                    seen_names.add(doc.metadata.get("Mekan Adı"))
            st.info(f"Genel İstanbul araması sonrası toplam {len(filtered_results)} sonuç var.")


        if not filtered_results:
            fallback_message = f"❌ Üzgünüm, **{location.capitalize()}** bölgesinde aradığınız kriterlere uygun bir mekan bulamadım."
            if location == "istanbul":
                fallback_message += " Veritabanımızda genel olarak İstanbul'da bu kriterlere uygun bir mekan bulamadım."
            else:
                fallback_message += " Belki aradığınız konumdaki verilerimiz eksiktir veya o bölgede kriterlerinize uyan bir yer yoktur. Lütfen farklı bir bölge veya daha genel bir arama yapmayı deneyin."

            sanitized_fallback_message = sanitize_markdown(fallback_message)
            state["messages"].append(AIMessage(content=sanitized_fallback_message))
            st.warning("Mekan bulunamadı, düşüş mesajı eklendi.")
            return state

        # Sadece ilk 5 sonucu göster
        formatted_results = []
        for doc in filtered_results[:5]: # İlk 5 sonuçla sınırla
            metadata = doc.metadata
            name = metadata.get("Mekan Adı", "Bilinmeyen Mekan")
            rating = float(metadata.get("Google Puanı", 0.0))
            review_count = int(metadata.get("Google Yorum Sayısı", 0))
            address = metadata.get("Adres", "Bilinmiyor")
            map_link = metadata.get("Maps Linki", "")
            phone = metadata.get("Telefon", "Yok")
            price_level_raw = metadata.get("Fiyat Seviyesi", "Yok")

            rating_display = f"⭐ {rating:.1f}" if rating > 0 else "⭐ Değerlendirilmemiş"
            review_count_display = f"({review_count} yorum)" if review_count > 0 else "(Yorum yok)"
            phone_display = f"📞 Telefon: {phone}" if phone and phone != "Yok" else ""

            price_display = ""
            if isinstance(price_level_raw, (int, float)):
                price_display = f"💸 Fiyat Seviyesi: {'₺' * int(price_level_raw)}"
            elif isinstance(price_level_raw, str) and price_level_raw.isdigit():
                 price_display = f"💸 Fiyat Seviyesi: {'₺' * int(price_level_raw)}"
            elif isinstance(price_level_raw, str) and price_level_raw != "Yok":
                 price_display = f"💸 Fiyat Seviyesi: {price_level_raw}"

            result_entry = (
                f"🏠 **{name}**\n"
                f"{rating_display} {review_count_display}\n"
                f"📍 {address}\n"
                f"{phone_display}\n"
                f"{price_display}\n"
                f"🔗 {map_link if map_link else 'Harita linki yok'}\n"
                "――――――――――――――――――――"
            )
            formatted_results.append(result_entry)

        sanitized_content = sanitize_markdown("Harika mekan önerilerim var:\n\n" + "\n".join(formatted_results))
        state["messages"].append(AIMessage(content=sanitized_content))
        st.success("Mekan arama sonuçları AIMessage olarak eklendi.")
        return state
    except Exception as e:
        st.error(f"⚠️ Arama sırasında beklenmedik bir hata oluştu (search_meyhaneler_node): {str(e)}")
        print(f"DEBUG ERROR in search_meyhaneler_node: {e}")
        sanitized_error_message = sanitize_markdown(f"⚠️ Arama sırasında beklenmedik bir hata oluştu: {str(e)}")
        state["messages"].append(AIMessage(content=sanitized_error_message))
        return state

def router_node(state: AgentState) -> AgentState:
    st.info("`router_node` çalışıyor.")
    last_msg = state["messages"][-1]
    content = last_msg.content.lower()
    st.info(f"Yönlendirme için son mesaj içeriği: {content}")

    # Use a flag for greeting to decide next_node
    greeting_triggers = ["selam", "merhaba", "günaydın", "naber", "nasılsın", "hi", "alo", "hey", "slm", "heyatım", "hayatım"]
    if any(g in content for g in greeting_triggers):
        state["next_node"] = "greeting" # New custom next_node for greetings
        st.info("Yönlendirme: greeting (selamlama)")
    elif any(t in content for t in ["meyhane", "restoran", "kafe", "date", "randevu", "mekan", "öneri", "neresi", "yer", "yemek", "içki"]):
        state["next_node"] = "search"
        st.info("Yönlendirme: search (mekan araması)")
    elif any(t in content for t in ["hava", "weather", "sıcaklık", "nem", "yağmur", "açık", "kapalı", "derece"]):
        state["next_node"] = "weather"
        st.info("Yönlendirme: weather (hava durumu)")
    elif any(t in content for t in ["fun fact", "ilginç bilgi", "bilgi ver", "biliyor muydun", "merak", "gerçek"]):
        state["next_node"] = "fun_fact"
        st.info("Yönlendirme: fun_fact (ilginç bilgi)")
    else:
        state["next_node"] = "general"
        st.info("Yönlendirme: general (genel yanıt)")
        
    return state


def fun_fact_node(state: AgentState) -> AgentState:
    st.info("`fun_fact_node` çalışıyor.")
    try:
        fact = get_fun_fact()
        print(f"DEBUG: Fun fact retrieved: {fact}")
        sanitized_fact = sanitize_markdown(f"🤔 İlginç Bilgi: {fact}")
        state["messages"].append(AIMessage(content=sanitized_fact))
        st.success("İlginç bilgi AIMessage olarak eklendi.")
    except Exception as e:
        st.error(f"ERROR in fun_fact_node: {e}")
        error_msg = "⚠️ İlginç bilgi alınırken bir hata oluştu"
        state["messages"].append(AIMessage(content=sanitize_markdown(error_msg)))
        st.warning("İlginç bilgi hatası AIMessage olarak eklendi.")
    return state

    
def general_response_node(state: AgentState) -> AgentState:
    st.info("`general_response_node` çalışıyor.")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)

    # Get the last human message
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        st.warning("general_response_node: İnsan mesajı bulunamadı.")
        # If no human message, return state and let the graph handle it
        state["next_node"] = "end_conversation_due_to_no_human_message" # A custom state for this edge case
        return state

    last_query = human_messages[-1].content.lower()
    st.info(f"General response için son kullanıcı sorgusu: {last_query}")

    # Greeting triggers (expanded list) - This part is now handled by the router
    # But as a fallback/redundancy, we can still have a basic check here.
    greeting_triggers = ["selam", "merhaba", "günaydın", "naber", "nasılsın", "hi", "alo", "hey", "slm", "heyatım", "hayatım"]
    is_greeting = any(g in last_query for g in greeting_triggers)

    if is_greeting:
        responses = [
            "Merhaba! 👋 İstanbul'da romantik mekan, meyhane, restoran ya da kafe önerisi almak ister misin?",
            "Selam! Size nasıl yardımcı olabilirim? Hava durumu bilgisi veya mekan önerisi alabilirsiniz. 🏙️",
            "Günaydın! ☀️ Hangi mekan ya da hava durumu bilgisiyle yardımcı olayım?",
            "Nasılsın? İstanbul'da nereye gitmek istersin? Romantik bir mekan mı, meyhane mi? 🍷"
        ]
        chosen = random.choice(responses)
        state["messages"].append(AIMessage(content=sanitize_markdown(chosen)))
        # For greetings, we can directly set next_node to indicate completion
        state["next_node"] = "greeting_handled" # Indicate that a greeting was handled and can end
        st.success("Selamlama yanıtı AIMessage olarak eklendi.")
        return state

    # If not a greeting, proceed with LLM
    try:
        st.info("LLM çağrısı yapılıyor...")
        response = llm.invoke(state["messages"])
        if response and response.content:
            st.success("LLM yanıtı alındı.")
            state["messages"].append(AIMessage(content=sanitize_markdown(response.content)))
            state["next_node"] = "general_handled" # Indicate that a general response was handled
        else:
            st.warning("LLM'den boş veya geçersiz yanıt alındı.")
            fallback = "Merhaba! Size nasıl yardımcı olabilirim?"
            state["messages"].append(AIMessage(content=sanitize_markdown(fallback)))
            state["next_node"] = "general_handled" # Fallback, then indicate handled
    except Exception as e:
        st.error(f"General response LLM çağrısında hata oluştu: {str(e)}")
        error_msg = f"Üzgünüm, bir hata oluştu: {str(e)}. Lütfen tekrar deneyin."
        state["messages"].append(AIMessage(content=sanitize_markdown(error_msg)))
        state["next_node"] = "general_handled" # On error, indicate handled

    return state  # CRITICAL: Return state after processing

@st.cache_resource
def create_workflow():
    st.info("LangGraph iş akışı oluşturuluyor...")
    workflow = StateGraph(AgentState)
    workflow.add_node("add_system_message", add_system_message)
    workflow.add_node("router", router_node)
    workflow.add_node("search", search_meyhaneler_node)
    workflow.add_node("weather", check_weather_node)
    workflow.add_node("general", general_response_node)
    workflow.add_node("fun_fact", fun_fact_node)
    workflow.add_node("summarize", summarize_conversation)

    workflow.add_edge(START, "add_system_message")
    workflow.add_edge("add_system_message", "router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_node"],
        {
            "search": "search",
            "weather": "weather",
            "general": "general",
            "fun_fact": "fun_fact",
            "greeting": "general", # Router will send greetings to general node
        }
    )

    # All specific nodes (search, weather, fun_fact) should now go to END
    workflow.add_edge("search", END)
    workflow.add_edge("weather", END)
    workflow.add_edge("fun_fact", END)

    # General node now explicitly handles its own ending based on the 'next_node' it sets
    workflow.add_conditional_edges(
        "general",
        lambda state: state.get("next_node", "general_handled"), # Default to general_handled if not set
        {
            "greeting_handled": END, # If it was a greeting, end
            "general_handled": END,  # If it was a general LLM response, end
            "end_conversation_due_to_no_human_message": END # If no human message, end
        }
    )

    # The summarize node always ends the turn for now. If you want more complex summarization flow, adjust this.
    workflow.add_edge("summarize", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    st.success("LangGraph iş akışı başarıyla derlendi.")

    return graph

# --- STREAMLIT UYGULAMASI ---
st.set_page_config(page_title="The Light Passanger 💬", page_icon="🌃")

# In your Streamlit UI code:
st.title("İstanbul Mekan Asistanı 💬")
st.markdown(sanitize_markdown(
    "Merhaba! Ben İstanbul'daki romantik mekan, meyhane, restoran ve kafe önerileri sunan yapay zeka asistanıyım. "
    "Size nasıl yardımcı olabilirim? 😊\n\n"
    "Örnek sorular:\n"
    "- `Selam! Beşiktaş'ta romantik bir mekan önerebilir misin?`\n"
    "- `Kadıköy'de hava durumu nasıl?`\n"
    "- `Bana ilginç bir bilgi verir misin?`"
))
# API Anahtarlarının ayarlı olup olmadığını kontrol et
if not st.secrets.get("OPENAI_API_KEY") or not st.secrets.get("OPENWEATHER_API_KEY") or not st.secrets.get("NEO4J_URI"):
    st.error("⚠️ Gerekli API anahtarları veya Neo4j bağlantı bilgileri eksik! Lütfen Streamlit Cloud kontrol panelinizdeki 'Secrets' kısmında `OPENAI_API_KEY`, `OPENWEATHER_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` değişkenlerini ayarlayın.")
    st.stop()
else:
    st.info("Tüm gerekli sırlar yüklendi.")


# LangGraph'ı başlat (sadece bir kez)
if "graph" not in st.session_state:
    st.session_state.graph = create_workflow()
if "conversation_thread_id" not in st.session_state:
    st.session_state.conversation_thread_id = str(uuid.uuid4())
    st.info(f"Yeni konuşma iş parçacığı ID'si: {st.session_state.conversation_thread_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.info("Mesaj geçmişi başlatıldı.")

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(sanitize_markdown(msg["content"]))

# Kullanıcıdan girdi al
if prompt := st.chat_input("Mesajınızı buraya yazın...", key="my_chat_input"):
    st.info(f"Kullanıcı girdisi: {prompt}")
    # Kullanıcı mesajını geçmişe ekle ve görüntüle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)  # Kullanıcı girdisini sanitize etmeyin

    # LangGraph için mesajları hazırla
    langgraph_messages = []
    # Add SystemMessage at the beginning of LangGraph messages for each run if not already present
    # This ensures the LLM always has the initial system prompt
    if not any(isinstance(msg, SystemMessage) for msg in st.session_state.messages):
         langgraph_messages.append(SystemMessage(content=SYSTEM_PROMPT))

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langgraph_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langgraph_messages.append(AIMessage(content=msg["content"]))
    st.info(f"LangGraph'a gönderilen toplam mesaj sayısı: {len(langgraph_messages)}")


    # LangGraph state'i oluştur
    current_state = {
        "messages": langgraph_messages,
        "last_recommended_place": None,
        "next_node": None,
        "location_query": None
    }
    st.info(f"Başlangıç LangGraph state'i: {current_state}")

    thread_id = st.session_state.conversation_thread_id

    with st.spinner("Düşünüyorum... 🤔"):
        try:
            final_state_data = {} # Initialize an empty dict to store the final state
            latest_ai_message_content = None

            # LangGraph akışını başlat ve tüm çıktıları işle
            for chunk in st.session_state.graph.stream(
                current_state,
                config={"configurable": {"thread_id": thread_id}}
            ):
                st.info(f"LangGraph adım sonucu: {chunk}")
                
                # If the chunk contains the final state, update final_state_data
                if "__end__" in chunk:
                    final_state_data = chunk["__end__"]
                # Otherwise, if it's an intermediate state with messages, update messages in final_state_data
                elif "messages" in chunk and chunk["messages"]:
                    final_state_data["messages"] = chunk["messages"]

            # After the stream, check the collected final_state_data
            if final_state_data and "messages" in final_state_data and final_state_data["messages"]:
                # Find the last AIMessage in the final collected state
                for msg in reversed(final_state_data["messages"]):
                    if isinstance(msg, AIMessage):
                        latest_ai_message_content = msg.content
                        break

                if latest_ai_message_content:
                    sanitized_content = sanitize_markdown(latest_ai_message_content)
                    st.session_state.messages.append({"role": "assistant", "content": sanitized_content})
                    st.success("Asistan yanıtı başarıyla eklendi.")
                    
                    with st.chat_message("assistant"):
                        st.markdown(sanitized_content, unsafe_allow_html=True)
                else:
                    error_msg = "Üzgünüm, bir yanıt üretemedim. LangGraph akışı tamamlandı ancak AI mesajı bulunamadı. Lütfen tekrar deneyin."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    with st.chat_message("assistant"):
                        st.markdown(error_msg)
                    st.error("LangGraph akışı AI mesajı üretmeden tamamlandı.")
            else:
                error_msg = "Üzgünüm, bir yanıt üretemedim. LangGraph akışı boş veya geçersiz bir durumla tamamlandı."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.error("LangGraph akışı boş veya geçersiz bir durumla tamamlandı.")

        except Exception as e:
            error_message = f"Bir hata oluştu: {e}. Lütfen daha sonra tekrar deneyin."
            st.error(f"Ana döngüde beklenmedik hata: {str(e)}")
            print(f"ERROR in main loop: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)
                st.exception(e)   # Rerun the app to show the latest message
    st.rerun()