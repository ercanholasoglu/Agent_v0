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

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    URI = st.secrets["NEO4J_URI"]
    USERNAME = st.secrets["NEO4J_USER"]
    PASSWORD = st.secrets["NEO4J_PASSWORD"]
    NEO4J_DATABASE = st.secrets.get("NEO4J_DATABASE", "neo4j")

except KeyError as e:
    st.error(f"Eksik Streamlit sırrı: {e}. Lütfen Streamlit Cloud kontrol panelinizde yapılandırın.")
    st.stop()


def sanitize_markdown(text):
    if not isinstance(text, str):

        return str(text)
    if not text:
        return ""

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    markdown_chars = ['\\', '*', '_', '~', '`', '#', '[', ']', '(', ')', '{', '}', '!', '^']
    for char in markdown_chars:
        text = text.replace(char, f"\\{char}")

    return text


def safe_markdown(text):
    try:

        re.compile(text)
        return text
    except re.error:

        return f"<pre>{text}</pre>"


class Neo4jConnector:
    def __init__(self):

        self.uri = st.secrets["NEO4J_URI"] 
        self.user = st.secrets["NEO4J_USER"] 
        self.password = st.secrets["NEO4J_PASSWORD"] 
        self.database = st.secrets.get("NEO4J_DATABASE", "neo4j")
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

@st.cache_resource
def initialize_retriever():
    meyhaneler_listesi = []
    try:
        conn = Neo4jConnector()
        meyhaneler_listesi = conn.get_meyhaneler(limit=10000)
        conn.close()
        if not meyhaneler_listesi:
            st.warning("Uyarı: Neo4j'den hiç mekan verisi çekilemedi. Dummy veri kullanılıyor.")
            meyhaneler_listesi = [
                {"name": "Dummy Meyhane A", "address": "Dummy Adres A", "rating": 4.0, "review_count": 100, "map_link": "http://dummy.map.a", "phone": "000", "price_level": 2, "neo4j_element_id": "dummy-a"},
                {"name": "Dummy Meyhane B", "address": "Dummy Adres B", "rating": 4.5, "review_count": 250, "map_link": "http://dummy.map.b", "phone": "000", "price_level": 3, "neo4j_element_id": "dummy-b"},
                {"name": "Dummy Meyhane C", "address": "Dummy Adres C", "rating": 3.8, "review_count": 50, "map_link": "http://dummy.map.c", "phone": "000", "price_level": 1, "neo4j_element_id": "dummy-c"},
            ]

        else:
            pass
    except Exception as e:
        st.error(f"Neo4j'den veri çekerken hata oluştu: {e}. Lütfen Neo4j sunucunuzun çalıştığından ve kimlik bilgilerinin doğru olduğundan emin olun. Dummy veri kullanılıyor.")
        meyhaneler_listesi = [
            {"name": "Dummy Meyhane A", "address": "Dummy Adres A", "rating": 4.0, "review_count": 100, "map_link": "http://dummy.map.a", "phone": "000", "price_level": 2, "neo4j_element_id": "dummy-a"},
            {"name": "Dummy Meyhane B", "address": "Dummy Adres B", "rating": 4.5, "review_count": 250, "map_link": "http://dummy.map.b", "phone": "000", "price_level": 3, "neo4j_element_id": "dummy-b"},
            {"name": "Dummy Meyhane C", "address": "Dummy Mekan C", "rating": 3.8, "review_count": 50, "map_link": "http://dummy.map.c", "phone": "000", "price_level": 1, "neo4j_element_id": "dummy-c"},
        ]


    processed_docs = process_documents(meyhaneler_listesi)
    try:
        vectorstore = InMemoryVectorStore.from_documents(
            documents=processed_docs,
            embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) 
        )
        st.success("Vektör deposu başarıyla oluşturuldu.")
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Vektör deposu oluşturulurken hata oluştu: {e}. OpenAI API anahtarınızı kontrol edin.")
        class DummyRetriever:
            def invoke(self, query, k):
                st.warning("Dummy retriever kullanılıyor. Gerçek arama yapılamıyor.")
                return [Document(page_content="Dummy Mekan", metadata={"Mekan Adı": "Dummy Mekan", "Adres": "Bilinmiyor", "Google Puanı": "0.0", "Google Yorum Sayısı": "0", "Maps Linki": "", "Telefon": "", "Fiyat Seviyesi": ""})]
        return DummyRetriever()

retriever = initialize_retriever()


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

@cached(TTLCache(maxsize=100, ttl=3600))
def get_fun_fact() -> str:
    try:
        response = requests.get("https://uselessfacts.jsph.pl/api/v2/facts/random?language=tr", timeout=5)
        response.raise_for_status() 
        fact = response.json().get("text", "İlginç bir bilgi bulunamadı.")
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
        weather_response = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=tr",
            timeout=10,
        )
        weather_response.raise_for_status()
        weather = weather_response.json()
        return weather
    except requests.exceptions.RequestException as e:
        st.error(f"OpenWeather API hatası: {e}")
        return {"error": f"API hatası: {e}"}
    except Exception as e:
        st.error(f"Hava durumu verisi alınırken beklenmedik hata: {e}")
        return {"error": f"Beklenmedik bir hata oluştu: {e}"}

def format_weather_response(location: str, data: Dict) -> str:
    if "error" in data:
        st.error(f"Hava durumu formatlama hatası: {data['error']}")
        return f"❌ {data['error']}"
    try:
        lines = [f"🌤️ **{location.capitalize()} Hava Durumu Tahmini:**"]
        if "list" not in data or not data["list"]:
            st.warning("Hava durumu verisi eksik veya boş.")
            return f"❌ {location} için hava durumu verisi bulunamadı."

        today = datetime.now().date()
        daily_forecasts = {}

        for item in data["list"]:
            timestamp = item["dt"]
            forecast_time = datetime.fromtimestamp(timestamp)
            forecast_date = forecast_time.date()

            # Sadece bugünden sonraki veya bugünü kapsayan 5 günlük tahmini al
            if forecast_date >= today and len(daily_forecasts) < 5:
                date_str = "Bugün" if forecast_date == today else forecast_time.strftime("%d %B")
                temp = item["main"]["temp"]
                description = item["weather"][0]["description"]
                icon_code = item["weather"][0]["icon"]
                icon_url = f"http://openweathermap.org/img/wn/{icon_code}.png"

                if forecast_date not in daily_forecasts:
                    daily_forecasts[forecast_date] = {
                        "date_str": date_str,
                        "temps": [],
                        "descriptions": set(),
                        "icons": set()
                    }
                daily_forecasts[forecast_date]["temps"].append(temp)
                daily_forecasts[forecast_date]["descriptions"].add(description)
                daily_forecasts[forecast_date]["icons"].add(icon_url)

        for date, forecast in daily_forecasts.items():
            avg_temp = sum(forecast["temps"]) / len(forecast["temps"])
            descriptions = ", ".join(list(forecast["descriptions"]))
            lines.append(
                f"- **{forecast['date_str']}**: Sıcaklık: {avg_temp:.1f}°C, Durum: {descriptions.capitalize()}."
            )

        return "\n".join(lines)
    except Exception as e:
        st.error(f"Hava durumu yanıtı formatlanırken beklenmedik hata: {e}")
        return f"❌ Hava durumu bilgisi formatlanırken bir hata oluştu: {e}"

class Tools:
    def __init__(self, llm_model, retriever):
        self.llm_model = llm_model
        self.retriever = retriever


    def search_places(self, state: AgentState) -> List[Document]:
        st.info("Mekan arama aracı çağrıldı...")
        messages = state['messages']
        query = messages[-1].content
        clean_query = clean_location_query(query)

        retrieved_docs = self.retriever.invoke(clean_query, k=5)
        st.success(f"{len(retrieved_docs)} mekan bulundu.")
        return retrieved_docs

    def get_weather_forecast(self, state: AgentState) -> str:
        st.info("Hava durumu aracı çağrıldı...")
        messages = state['messages']
        query = messages[-1].content
        location = clean_location_query(query) 
        
        weather_data = get_openweather_forecast(location)
        formatted_weather = format_weather_response(location, weather_data)
        st.success(f"{location} için hava durumu bilgisi çekildi.")
        return formatted_weather

    def provide_fun_fact(self, state: AgentState) -> str:
        st.info("İlginç bilgi aracı çağrıldı...")
        return get_fun_fact()

    def generate_response(self, state: AgentState) -> str:
        st.info("Yanıt oluşturma aracı çağrıldı...")
        messages = state['messages']

        user_message_content = messages[-1].content
        retrieved_docs = self.retriever.invoke(user_message_content, k=5) 

        context_str = "\n".join([doc.page_content for doc in retrieved_docs])
        if context_str:
            st.info(f"Yanıt oluşturmak için {len(retrieved_docs)} doküman kullanılıyor.")
            full_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT + "\n\nKullanılabilecek ek bilgi/mekanlar:\n" + context_str),
                MessagesPlaceholder(variable_name="messages"),
            ])
        else:
            st.warning("Mekan önerisi için uygun doküman bulunamadı, genel prompt kullanılıyor.")
            full_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
            ])
        
        try:
            response = self.llm_model.invoke(full_prompt.format_messages(messages=messages))
            st.success("Yanıt başarıyla oluşturuldu.")
            return response.content
        except Exception as e:
            st.error(f"Yanıt oluşturulurken hata oluştu: {e}")
            return "Üzgünüm, şu an bir yanıt oluşturamıyorum."

    def summarize_conversation(self, state: AgentState) -> str:
        st.info("Sohbet özetleme aracı çağrıldı...")
        messages = state['messages']
        recent_messages = messages[-5:]
        try:
            summary_response = self.llm_model.invoke(SUMMARY_PROMPT.format_messages(messages=recent_messages))
            st.success("Sohbet özeti oluşturuldu.")
            return summary_response.content
        except Exception as e:
            st.error(f"Sohbet özetlenirken hata oluştu: {e}")
            return "Sohbet özetlenemedi."

    def route_question(self, state: AgentState) -> str:
        st.info("Soru yönlendirme aracı çağrıldı...")
        messages = state["messages"]
        last_message = messages[-1]

        if "hava" in last_message.content.lower() or "sıcaklık" in last_message.content.lower():
            st.success("Hava durumu rotasına yönlendiriliyor.")
            return "weather"
        if "ilginç bilgi" in last_message.content.lower() or "bilgi ver" in last_message.content.lower():
            st.success("İlginç bilgi rotasına yönlendiriliyor.")
            return "fun_fact"
        if "özetle" in last_message.content.lower() or "özet" in last_message.content.lower():
            st.success("Özetleme rotasına yönlendiriliyor.")
            return "summarize"
        
        place_keywords = ["mekan", "restoran", "kafe", "meyhane", "nereye gideyim", "öneri", "yer"]
        if any(keyword in last_message.content.lower() for keyword in place_keywords):
            st.success("Mekan arama rotasına yönlendiriliyor.")
            return "search_places"

        st.success("Genel yanıt rotasına yönlendiriliyor.")
        return "generate_response" 


def create_workflow():
    st.info("LangGraph iş akışı oluşturuluyor...")
    workflow = StateGraph(AgentState)

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    tools = Tools(llm_model=llm, retriever=retriever)

    workflow.add_node("route_question", tools.route_question)
    workflow.add_node("search_places", tools.search_places)
    workflow.add_node("get_weather_forecast", tools.get_weather_forecast)
    workflow.add_node("provide_fun_fact", tools.provide_fun_fact)
    workflow.add_node("generate_response", tools.generate_response)
    workflow.add_node("summarize_conversation", tools.summarize_conversation)

    workflow.set_entry_point("route_question")

    workflow.add_conditional_edges(
        "route_question",
        lambda state: state["next_node"] if state.get("next_node") else state["messages"][-1].content, # HACK: Use content for routing
        {
            "search_places": "search_places",
            "weather": "get_weather_forecast",
            "fun_fact": "provide_fun_fact",
            "summarize": "summarize_conversation",
            "generate_response": "generate_response",
        },
    )

    workflow.add_edge("search_places", "generate_response")
    workflow.add_edge("get_weather_forecast", "generate_response")
    workflow.add_edge("provide_fun_fact", "generate_response")
    workflow.add_edge("summarize_conversation", "generate_response")
    
    workflow.add_edge("generate_response", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    st.success("LangGraph iş akışı başarıyla oluşturuldu.")
    return app

# --- Streamlit Uygulaması ---
st.set_page_config(page_title="İstanbul Mekan Asistanı", layout="wide")
st.title("İstanbul Mekan ve Hava Durumu Asistanı 📍")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)


if prompt := st.chat_input("Nasıl yardımcı olabilirim?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Yanıt oluşturuluyor..."):
        app = create_workflow()
        

        if "conversation_thread_id" not in st.session_state:
            st.session_state.conversation_thread_id = str(uuid.uuid4())
            
        
        config = {"configurable": {"thread_id": st.session_state.conversation_thread_id}}

        response_placeholder = st.empty() 
        latest_ai_message_content = ""

        try:
            for s in app.stream({"messages": [HumanMessage(content=prompt)]}, config=config):
                for key in s:
                    node_output = s[key]
                    if "messages" in node_output:
                        for msg in reversed(node_output["messages"]):
                            if isinstance(msg, AIMessage) and msg.content:
                                latest_ai_message_content = msg.content
                                break 


 
            if latest_ai_message_content:
                sanitized_content = sanitize_markdown(latest_ai_message_content)
                st.session_state.messages.append({"role": "assistant", "content": sanitized_content})
                
                with st.chat_message("assistant"):
                    st.markdown(sanitized_content, unsafe_allow_html=True)
            else:
                error_msg = "Üzgünüm, bir yanıt üretemedim. LangGraph akışı tamamlandı ancak AI mesajı bulunamadı. Lütfen tekrar deneyin."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.error("LangGraph akışı AI mesajı üretmeden tamamlandı.")

        except Exception as e:
            error_message = f"Bir hata oluştu: {e}. Lütfen daha sonra tekrar deneyin."
            st.error(f"Ana döngüde beklenmedik hata: {str(e)}")
            print(f"ERROR in main loop: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)
                st.exception(e)  
    st.rerun()