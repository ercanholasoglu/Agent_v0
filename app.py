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
##from dotenv import load_dotenv
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


try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"] # If you use Tavily, ensure it's in secrets
    URI = st.secrets["NEO4J_URI"]
    USERNAME = st.secrets["NEO4J_USER"]
    PASSWORD = st.secrets["NEO4J_PASSWORD"]
    NEO4J_DATABASE = st.secrets.get("NEO4J_DATABASE", "neo4j") # Use .get with a default for optional secrets
except KeyError as e:
    st.error(f"Missing Streamlit secret: {e}. Please configure this in your Streamlit Cloud dashboard.")
    st.stop() # Stop the app if essential secrets are missing
    
    
def sanitize_markdown(text):
    if not isinstance(text, str):
        return str(text)
    if not text:
        return ""
    
    # HTML özel karakterleri
    text = text.replace("&", "&amp;").replace("<", "<").replace(">", ">")
    
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
        # These are correctly reading from NEO4J_URI, NEO4J_USER, etc.
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = None
        
    def connect(self):
        """Establishes a connection to Neo4j."""
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                with self.driver.session(database=self.database) as session:
                    session.run("RETURN 1")
            except Exception as exc:
                st.error(f"Neo4j bağlantı hatası: {exc}")
                raise ConnectionError(f"Neo4j bağlantı hatası: {exc}") from exc

    def close(self):
        """Closes the Neo4j driver if it's open."""
        if self.driver:
            self.driver.close()
            self.driver = None
            
    if not st.secrets.get("OPENAI_API_KEY") or not st.secrets.get("OPENWEATHER_API_KEY"):
        st.error("⚠️ API anahtarları eksik! Lütfen Streamlit Cloud kontrol panelinizdeki 'Secrets' kısmında `OPENAI_API_KEY` ve `OPENWEATHER_API_KEY` değişkenlerini ayarlayın.")
        st.stop()

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
                return [self._clean_record(record) for record in result]
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
    return processed

# --- Neo4j Verilerini Yükle ve Vektör Deposunu Oluştur (Sadece Bir Kez Çalıştır) ---
@st.cache_resource
def initialize_retriever():
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
    except Exception as e:
        st.error(f"Neo4j'den veri çekerken hata oluştu: {e}. Lütfen Neo4j sunucunuzun çalıştığından ve kimlik bilgilerinin doğru olduğundan emin olun. Dummy veri kullanılıyor.")
        meyhaneler_listesi = [
            {"name": "Dummy Meyhane A", "address": "Dummy Adres A", "rating": 4.0, "review_count": 100, "map_link": "http://dummy.map.a", "phone": "000", "price_level": 2, "neo4j_element_id": "dummy-a"},
            {"name": "Dummy Meyhane B", "address": "Dummy Adres B", "rating": 4.5, "review_count": 250, "map_link": "http://dummy.map.b", "phone": "000", "price_level": 3, "neo4j_element_id": "dummy-b"},
            {"name": "Dummy Meyhane C", "address": "Dummy Adres C", "rating": 3.8, "review_count": 50, "map_link": "http://dummy.map.c", "phone": "000", "price_level": 1, "neo4j_element_id": "dummy-c"},
        ]
    
    processed_docs = process_documents(meyhaneler_listesi)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=processed_docs,
        embedding=OpenAIEmbeddings()
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})

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
    try:
        response = requests.get("https://uselessfacts.jsph.pl/api/v2/facts/random?language=tr", timeout=5)
        if response.status_code == 200:
            return response.json().get("text", "İlginç bir bilgi bulunamadı.")
        return "Bugün için ilginç bir bilgi yok."
    except Exception:
        return "İlginç bilgi servisi şu an çalışmıyor."

def clean_location_query(query: str) -> str:
    normalized_query = unicodedata.normalize('NFKD', query.lower()).encode('ascii', 'ignore').decode('utf-8')

    istanbul_locations = [
        r'etiler', r'levent', r'maslak', r'nisantasi', r'nisantaşi', 
        r'bebek', r'arnavutkoy', r'arnavutköy', r'ortakoy', r'ortaköy', r'cihangir',
        r'taksim', r'karakoy', r'karaköy', r'galata', r'fatih',
        r'sultanahmet', r'eminonu', r'eminönü', r'kadikoy', r'kadıköy', r'moda',
        r'bagdat caddesi', r'bağdat caddesi', r'suadiye', r'bostanci', r'bostancı',
        r'maltepe', r'kartal', r'pendik', r'uskudar', r'üsküdar',
        r'camlica', r'çamlıca', r'beykoz', r'atasehir', r'ataşehir', r'cekmekoy', r'çekmeköy',
        r'sariyer', r'sarıyer', r'istinye', r'tarabya', r'yenikoy', r'yeniköy',
        r'bahcekoy', r'bahçeköy', r'buyukdere', r'büyükdere', r'zumrutevler', r'zümrütevler',
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
            return match.group(0)

    general_cities = [
        r'istanbul', r'ankara', r'izmir', r'adana',
        r'bursa', r'antalya', r'konya', r'kayseri',
        r'gaziantep', r'samsun', r'eskisehir', r'eskişehir', r'duzce', r'düzce'
    ]

    for city_regex in general_cities:
        match = re.search(r'\b' + city_regex + r'\b', normalized_query)
        if match:
            return match.group(0)

    return "istanbul"

weather_cache = TTLCache(maxsize=100, ttl=300)

@cached(weather_cache)
def get_openweather_forecast(location: str) -> Dict:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "API anahtarı bulunamadı."}
    try:
        geo = requests.get(
            f"http://api.openweathermap.org/geo/1.0/direct?q={location},TR&limit=1&appid={api_key}",
            timeout=10,
        ).json()
        if not geo:
            return {"error": f"'{location}' konumu bulunamadı."}
        lat, lon = geo[0]["lat"], geo[0]["lon"]
        weather = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=tr",
            timeout=10,
        ).json()
        return weather
    except requests.exceptions.RequestException as e:
        return {"error": f"API hatası: {e}"}

def format_weather_response(location: str, data: Dict) -> str:
    if "error" in data:
        return f"❌ {data['error']}"
    try:
        lines = [f"🌤️ **{location.capitalize()} Hava Durumu Tahmini:**"]
        if "list" not in data or not data["list"]:
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
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Hava durumu verisi işlenirken hata oluştu: {e}"

# --- LangGraph Nodes ---
def check_weather_node(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]
    query = last_msg.content

    location = state.get("location_query") or clean_location_query(query)
    state["location_query"] = location 

    formatted = format_weather_response(location, get_openweather_forecast(location))

    sanitized_formatted = sanitize_markdown(formatted)
    state["messages"].append(AIMessage(content=sanitized_formatted))
    return state

def add_system_message(state: AgentState) -> AgentState:
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
        return {"messages": [system_msg] + state["messages"]}
    else:
        return state

def summarize_conversation(state: AgentState) -> AgentState:
    messages = state["messages"]

    if len(messages) > 5: # Konuşma belirli bir uzunluğu aşınca özetle
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        chain = SUMMARY_PROMPT | llm
        summary = chain.invoke({"messages": messages})

        return {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                AIMessage(content=summary.content) 
            ]
        }
    else:
        return state

def search_meyhaneler_node(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]
    query = last_msg.content

    location = clean_location_query(query)
    state["location_query"] = location 

    try:
        # Arama sorgusunu iyileştir
        retrieval_query = f"{location} {query}" if location and location not in query.lower() else query
        raw_results = retriever.invoke(retrieval_query, k=10) # Daha fazla sonuç alıp filtreleyelim
        filtered_results = []

        normalized_location = unicodedata.normalize('NFKD', location.lower()).encode('ascii', 'ignore').decode('utf-8')

        # Konuma göre filtreleme
        for doc in raw_results:
            address_lower = unicodedata.normalize('NFKD', doc.metadata.get("Adres", "").lower()).encode('ascii', 'ignore').decode('utf-8')
            name_lower = unicodedata.normalize('NFKD', doc.metadata.get("Mekan Adı", "").lower()).encode('ascii', 'ignore').decode('utf-8')
            
            # Eğer konum sorguda yer alıyorsa ve mekanın adresi veya adı bu konumu içeriyorsa ekle
            if normalized_location in address_lower or normalized_location in name_lower:
                filtered_results.append(doc)
        
        # Eğer filtrelemeden sonra hala sonuç yoksa veya ilk 3 sonuç yoksa daha geniş arama yap
        if not filtered_results or len(filtered_results) < 3:
            # Sadece İstanbul için genel arama
            raw_results_istanbul = retriever.invoke(f"istanbul {query}", k=5)
            # Daha önce filtrelenmiş sonuçları da ekle, mükerrerleri önle
            seen_names = {doc.metadata.get("Mekan Adı") for doc in filtered_results}
            for doc in raw_results_istanbul:
                if doc.metadata.get("Mekan Adı") not in seen_names:
                    filtered_results.append(doc)
                    seen_names.add(doc.metadata.get("Mekan Adı"))


        if not filtered_results:
            fallback_message = f"❌ Üzgünüm, **{location.capitalize()}** bölgesinde aradığınız kriterlere uygun bir mekan bulamadım."
            if location == "istanbul": 
                fallback_message += " Veritabanımızda genel olarak İstanbul'da bu kriterlere uygun bir mekan bulamadım."
            else: 
                fallback_message += " Belki aradığınız konumdaki verilerimiz eksiktir veya o bölgede kriterlerinize uyan bir yer yoktur. Lütfen farklı bir bölge veya daha genel bir arama yapmayı deneyin."
            
            sanitized_fallback_message = sanitize_markdown(fallback_message)
            state["messages"].append(AIMessage(content=sanitized_fallback_message))
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

        sanitized_content = sanitize_markdown("\n".join(formatted_results))
        state["messages"].append(AIMessage(content=sanitized_content))
        return state
    except Exception as e:
        print(f"DEBUG ERROR in search_meyhaneler_node: {e}")
        sanitized_error_message = sanitize_markdown(f"⚠️ Arama sırasında beklenmedik bir hata oluştu: {str(e)}")
        state["messages"].append(AIMessage(content=sanitized_error_message))
        return state

def router_node(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]
    content = last_msg.content.lower()
    
    if any(t in content for t in ["meyhane", "restoran", "kafe", "date", "randevu", "mekan", "öneri", "neresi", "yer", "yemek", "içki"]):
        state["next_node"] = "search"
    elif any(t in content for t in ["hava", "weather", "sıcaklık", "nem", "yağmur", "açık", "kapalı", "derece"]):
        state["next_node"] = "weather"
    elif any(t in content for t in ["fun fact", "ilginç bilgi", "bilgi ver", "biliyor muydun", "merak", "gerçek"]):
        state["next_node"] = "fun_fact"
    else:
        state["next_node"] = "general"
        print("DEBUG: Router node set next_node to 'general'") # Bu satırı ekleyin
    return state
def fun_fact_node(state: AgentState) -> AgentState:
    fact = get_fun_fact()

    sanitized_fact = sanitize_markdown(f"🤔 İlginç Bilgi: {fact}")
    state["messages"].append(AIMessage(content=sanitized_fact))
    return state

def general_response_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    last_query = human_messages[-1].content.lower() if human_messages else ""

    # Check for greeting triggers FIRST
    greeting_triggers = ["selam", "merhaba", "günaydın", "naber", "nasılsın", "hi", "alo"]
    if any(g in last_query for g in greeting_triggers):
        responses = [
            "Merhaba! 👋 İstanbul'da romantik mekan, meyhane, restoran ya da kafe önerisi almak ister misin?",
            "Selam! Size nasıl yardımcı olabilirim? Hava durumu bilgisi veya mekan önerisi alabilirsiniz. 🏙️",
            "Günaydın! ☀️ Hangi mekan ya da hava durumu bilgisiyle yardımcı olayım?",
            "Nasılsın? İstanbul'da nereye gitmek istersin? Romantik bir mekan mı, meyhane mi? 🍷"
        ]
        chosen = random.choice(responses)
        state["messages"].append(AIMessage(content=sanitize_markdown(chosen))) # Sanitize here
        return state # Return early if it's a greeting

    # If not a greeting, proceed with LLM invocation
    try:
        response = llm.invoke(state["messages"])
        if response and hasattr(response, "content") and response.content:
            sanitized_content = sanitize_markdown(response.content)
            state["messages"].append(AIMessage(content=sanitized_content))
        else:
            # Fallback if LLM returns empty/invalid
            sanitized_fallback = sanitize_markdown("Üzgünüm, bir yanıt üretemedim veya isteğinizi anlayamadım. Size nasıl yardımcı olabilirim?")
            state["messages"].append(AIMessage(content=sanitized_fallback))
    except Exception as e:
        # Catch any errors during LLM invocation
        error_message = f"⚠️ Hata: {str(e)}. LLM yanıtı alınamadı."
        print(f"DEBUG: Error in general_response_node LLM invocation: {e}")
        sanitized_error = sanitize_markdown(error_message)
        state["messages"].append(AIMessage(content=sanitized_error))
        # Add a general fallback if an error occurs
        sanitized_fallback_after_error = sanitize_markdown("Merhaba! Size nasıl yardımcı olabilirim?")
        state["messages"].append(AIMessage(content=sanitized_fallback_after_error))

    return state

@st.cache_resource
def create_workflow():
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
        }
    )

    workflow.add_edge("search", END)
    workflow.add_edge("weather", END)
    workflow.add_edge("fun_fact", END)
    
    workflow.add_conditional_edges(
        "general",
        lambda state: "summarize" if len(state["messages"]) > 5 else END, 
        {
            "summarize": "summarize", 
            END: END 
        }
    )
    workflow.add_edge("summarize", END) 

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph

# --- STREAMLIT UYGULAMASI ---
st.set_page_config(page_title="İstanbul Mekan Asistanı 💬", page_icon="🌃")

st.title("İstanbul Mekan Asistanı 💬")
st.markdown(sanitize_markdown("Merhaba! Ben İstanbul'daki romantik mekan, meyhane, restoran ve kafe önerileri sunan yapay zeka asistanıyım. Ayrıca hava durumu bilgisi veya ilginç bilgiler de sağlayabilirim. Size nasıl yardımcı olabilirim? 😊"))

# API Anahtarlarının ayarlı olup olmadığını kontrol et
# This check will now work correctly after load_dotenv() is at the top
if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENWEATHER_API_KEY"): # Changed to os.getenv as it's safer
    st.error("⚠️ API anahtarları eksik! Lütfen `os.environ` içinde `OPENAI_API_KEY` ve `OPENWEATHER_API_KEY` değişkenlerini ayarlayın.")
    st.stop() # Uygulamayı durdur

# LangGraph'ı başlat (sadece bir kez)
if "graph" not in st.session_state:
    st.session_state.graph = create_workflow()
if "conversation_thread_id" not in st.session_state:
    # Generate a unique thread ID for a new conversation or retrieve an existing one
    st.session_state.conversation_thread_id = str(uuid.uuid4()) # Use uuid for uniqueness

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(sanitize_markdown(msg["content"]))

# Kullanıcıdan girdi al (SADECE BURADA OLMALI)
if prompt := st.chat_input("Mesajınızı buraya yazın...", key="my_chat_input"):
    # Kullanıcı mesajını geçmişe ekle ve görüntüle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(sanitize_markdown(prompt))


    # LangGraph'ı çalıştırma ve yanıt üretme
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    thread_id = st.session_state.conversation_thread_id
    
    with st.spinner("Düşünüyorum... 🤔"):
        try:
            latest_ai_content = "" 
            for s in st.session_state.graph.stream(inputs, config={"configurable": {"thread_id": thread_id}}):
                 print(f"DEBUG: Stream step: {s}") # Add this to see all streamed output
                 if "__end__" not in s:
                    ai_response_message = s.get("messages", [])[-1] if s.get("messages") else None
                    if ai_response_message and isinstance(ai_response_message, AIMessage):
                        latest_ai_content = ai_response_message.content 
                        print(f"DEBUG: Latest AI content from stream: {latest_ai_content}") # See content as it's built

            if latest_ai_content:
                sanitized_final_ai_response = sanitize_markdown(latest_ai_content)
                print(f"DEBUG: Final AI response (sanitized): {sanitized_final_ai_response}")
            else:
                sanitized_final_ai_response = sanitize_markdown("Üzgünüm, bir yanıt üretemedim.")
                print("DEBUG: No AI content produced from stream.")

            st.session_state.messages.append({"role": "assistant", "content": sanitized_final_ai_response})
            with st.chat_message("assistant"):
                st.markdown(sanitized_final_ai_response) # st.html yerine st.markdown kullanmaya devam edin

        except Exception as e:
            error_message = f"Bir hata oluştu: {e}. Lütfen daha sonra tekrar deneyin veya farklı bir soru sorun."
            print(f"DEBUG: Main Streamlit loop exception: {e}") # Ana döngüdeki hataları yakalamak için
            sanitized_error_message = sanitize_markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": sanitized_error_message})
            with st.chat_message("assistant"):
                st.markdown(sanitized_error_message)
                st.exception(e) # Hatanın detaylarını Streamlit arayüzünde göster