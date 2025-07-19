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
except KeyError as e:
    st.error(f"Eksik Streamlit sırrı: {e}. Lütfen Streamlit Cloud kontrol panelinizde yapılandırın.")
    st.stop()


def sanitize_markdown(text):
    if not isinstance(text, str):
        return str(text)
    if not text:
        return ""

    # Basic HTML escaping for safety
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Escape markdown special characters
    markdown_chars = ['\\', '*', '_', '~', '`', '#', '[', ']', '(', ')', '{', '}', '!', '^']
    for char in markdown_chars:
        text = text.replace(char, f"\\{char}")

    return text

def safe_markdown(text):
    # This function seems redundant given sanitize_markdown,
    # and its regex compilation check is not suitable for general markdown.
    # It's better to rely on Streamlit's markdown rendering with safe text.
    return text # Assuming sanitize_markdown already handled safety

class Neo4jConnector:
    def __init__(self):
        self.uri = st.secrets["NEO4J_URI"]
        self.user = st.secrets["NEO4J_USER"]
        self.password = st.secrets["NEO4J_PASSWORD"]
        self.database = st.secrets.get("NEO4J_DATABASE", "neo4j")
        self.driver = None

    def connect(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                with self.driver.session(database=self.database) as session:
                    session.run("RETURN 1") # Test connection
            except Exception as exc:
                # st.error(f"Neo4j bağlantı hatası: {exc}") # Removed for pop-up reduction
                raise ConnectionError(f"Neo4j bağlantı hatası: {exc}") from exc

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None


    def get_meyhaneler(self, limit: int = 10000) -> List[Dict[str, Any]]:
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
            # st.error(f"Neo4j sorgu hatası (get_meyhaneler): {exc}") # Removed for pop-up reduction
            print(f"Sorgu hatası (get_meyhaneler): {exc}")
            return []

    @staticmethod
    def _clean_record(record) -> Dict[str, Any]:
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

# This add_messages is part of LangGraph's message history management, not a Streamlit display helper.
# It's correctly used with Annotated[List[BaseMessage], add_messages]
def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
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
            st.warning("Uyarı: Neo4j'den hiç mekan verisi çekilemedi. Dummy veri kullanılıyor.") # Keep this warning
            meyhaneler_listesi = [
                {"name": "Dummy Meyhane A", "address": "Dummy Adres A", "rating": 4.0, "review_count": 100, "map_link": "http://dummy.map.a", "phone": "000", "price_level": 2, "neo4j_element_id": "dummy-a"},
                {"name": "Dummy Meyhane B", "address": "Dummy Adres B", "rating": 4.5, "review_count": 250, "map_link": "http://dummy.map.b", "phone": "000", "price_level": 3, "neo4j_element_id": "dummy-b"},
                {"name": "Dummy Meyhane C", "address": "Dummy Adres C", "rating": 3.8, "review_count": 50, "map_link": "http://dummy.map.c", "phone": "000", "price_level": 1, "neo4j_element_id": "dummy-c"},
            ]

        else:
            pass
    except Exception as e:
        st.error(f"Neo4j'den veri çekerken hata oluştu: {e}. Lütfen Neo4j sunucunuzun çalıştığından ve kimlik bilgilerinin doğru olduğundan emin olun. Dummy veri kullanılıyor.") # Keep this error
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
        # st.success("Vektör deposu başarıyla oluşturuldu.") # Removed for pop-up reduction
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Vektör deposu oluşturulurken hata oluştu: {e}. OpenAI API anahtarınızı kontrol edin.") # Keep this error
        class DummyRetriever:
            def invoke(self, query, k):
                # st.warning("Dummy retriever kullanılıyor. Gerçek arama yapılamıyor.") # Removed for pop-up reduction
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
        # st.error("İlginç bilgi servisi zaman aşımına uğradı.") # Removed for pop-up reduction
        return "İlginç bilgi servisi şu an çok yavaş veya çalışmıyor."
    except requests.exceptions.RequestException as e:
        # st.error(f"İlginç bilgi servisi hatası: {e}") # Removed for pop-up reduction
        return f"İlginç bilgi servisi şu an çalışmıyor. Hata: {e}"
    except Exception as e:
        # st.error(f"İlginç bilgi alınırken beklenmedik hata: {e}") # Removed for pop-up reduction
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
    api_key = st.secrets.get("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "API anahtarı bulunamadı."}
    try:
        geo_response = requests.get(
            f"http://api.openweathermap.org/geo/1.0/direct?q={location},TR&limit=1&appid={api_key}",
            timeout=10,
        )
        geo_response.raise_for_status()
        geo = geo_response.json()
        if not geo:
            return {"error": f"'{location}' konumu bulunamadı."}
        
        # Hata burada! "_lon" yerine "lon" olmalı:
        lat, lon = geo[0]["lat"], geo[0]["lon"] # Düzeltilen Satır
        
        weather_response = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=tr",
            timeout=10,
        )
        weather_response.raise_for_status()
        weather = weather_response.json()
        return weather
    except requests.exceptions.RequestException as e:
        return {"error": f"API hatası: {e}"}
    except Exception as e:
        return {"error": f"Beklenmedik bir hata oluştu: {e}"}

def format_weather_response(location: str, data: Dict) -> str:
    if "error" in data:
        # st.error(f"Hava durumu formatlama hatası: {data['error']}") # Removed for pop-up reduction
        return f"❌ {data['error']}"
    try:
        lines = [f"🌤️ **{location.capitalize()} Hava Durumu Tahmini:**"]
        if "list" not in data or not data["list"]:
            # st.warning("Hava durumu verisi eksik veya boş.") # Removed for pop-up reduction
            return f"❌ {location} için hava durumu verisi bulunamadı."

        today = datetime.now().date()
        daily_forecasts = {}

        for item in data["list"]:
            timestamp = item["dt"]
            forecast_time = datetime.fromtimestamp(timestamp)
            forecast_date = forecast_time.date()

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
        # st.error(f"Hava durumu yanıtı formatlanırken beklenmedik hata: {e}") # Removed for pop-up reduction
        return f"❌ Hava durumu bilgisi formatlanırken bir hata oluştu: {e}"

class Tools:
    def __init__(self, llm_model, retriever):
        self.llm_model = llm_model
        self.retriever = retriever


    def search_places(self, state: AgentState) -> AgentState:
        # st.info("Mekan arama aracı çağrıldı...") # Removed for pop-up reduction
        messages = state['messages']
        # The query should be derived from the actual user's last message, not necessarily the content of the last message which could be a SystemMessage
        # We need to find the latest HumanMessage for the query
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query: # Fallback if no HumanMessage is found (unlikely in a conversation flow)
            user_query = messages[-1].content # Use the very last message content as fallback

        clean_query = clean_location_query(user_query)

        retrieved_docs = self.retriever.invoke(clean_query, k=5)
        # st.success(f"{len(retrieved_docs)} mekan bulundu.") # Removed for pop-up reduction
        
        formatted_docs = "\n".join([f"- Mekan Adı: {doc.metadata.get('Mekan Adı', 'Bilinmiyor')}, Adres: {doc.metadata.get('Adres', 'Bilinmiyor')}, Google Puanı: {doc.metadata.get('Google Puanı', '0.0')}, Yorum Sayısı: {doc.metadata.get('Google Yorum Sayısı', '0')}, Fiyat Seviyesi: {doc.metadata.get('Fiyat Seviyesi', 'Yok')}" for doc in retrieved_docs])
        tool_output_message = f"Bulunan mekanlar:\n{formatted_docs}"
        
        return {"messages": [SystemMessage(content=tool_output_message)]}

    def get_weather_forecast(self, state: AgentState) -> AgentState:
        # st.info("Hava durumu aracı çağrıldı...") # Removed for pop-up reduction
        messages = state['messages']
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        if not user_query:
            user_query = messages[-1].content

        location = clean_location_query(user_query)
        
        weather_data = get_openweather_forecast(location)
        formatted_weather = format_weather_response(location, weather_data)
        # st.success(f"{location} için hava durumu bilgisi çekildi.") # Removed for pop-up reduction
        
        return {"messages": [SystemMessage(content=formatted_weather)]}

    def provide_fun_fact(self, state: AgentState) -> AgentState:
        # st.info("İlginç bilgi aracı çağrıldı...") # Removed for pop-up reduction
        fun_fact = get_fun_fact()
        return {"messages": [SystemMessage(content=fun_fact)]}

    def generate_response(self, state: AgentState) -> AgentState:
        # st.info("Yanıt oluşturma aracı çağrıldı...") # Removed for pop-up reduction
        messages = state['messages']

        llm_messages = []
        llm_messages.append(SystemMessage(content=SYSTEM_PROMPT))
        llm_messages.extend(messages) # Add all messages from the state

        user_message_content = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message_content = msg.content
                break
        if not user_message_content:
            user_message_content = messages[-1].content # Fallback

        retrieved_docs = self.retriever.invoke(user_message_content, k=5)

        context_str = "\n".join([doc.page_content for doc in retrieved_docs])
        
        if context_str:
            llm_messages.append(SystemMessage(content=f"Kullanılabilecek ek bilgi/mekanlar:\n{context_str}"))
            # st.info(f"Yanıt oluşturmak için {len(retrieved_docs)} doküman kullanılıyor.") # Removed for pop-up reduction
        else:
            # st.warning("Mekan önerisi için uygun doküman bulunamadı, genel prompt kullanılıyor.") # Removed for pop-up reduction
            pass # No warning if no docs found, just proceed with general prompt
        
        # The prompt for LLM should include the full conversation history.
        # We need to make sure the messages passed to the LLM are the ones from AgentState,
        # and that the SystemMessage is prepended correctly.
        # The full_prompt template already includes MessagesPlaceholder.
        
        try:
            # The full_prompt is just MessagesPlaceholder, so it directly uses the messages list.
            response = self.llm_model.invoke(llm_messages) # Pass the list of messages directly
            # st.success("Yanıt başarıyla oluşturuldu.") # Removed for pop-up reduction
            return {"messages": [AIMessage(content=response.content)]}
        except Exception as e:
            # st.error(f"Yanıt oluşturulurken hata oluştu: {e}") # Removed for pop-up reduction
            return {"messages": [AIMessage(content="Üzgünüm, şu an bir yanıt oluşturamıyorum.")]}


    def summarize_conversation(self, state: AgentState) -> AgentState:
        # st.info("Sohbet özetleme aracı çağrıldı...") # Removed for pop-up reduction
        messages = state['messages']
        recent_messages = messages[-5:] # Summarize last 5 messages
        try:
            summary_response = self.llm_model.invoke(SUMMARY_PROMPT.format_messages(messages=recent_messages))
            # st.success("Sohbet özeti oluşturuldu.") # Removed for pop-up reduction
            return {"messages": [SystemMessage(content=summary_response.content)]}
        except Exception as e:
            # st.error(f"Sohbet özetlenirken hata oluştu: {e}") # Removed for pop-up reduction
            return {"messages": [SystemMessage(content="Sohbet özetlenemedi.")]}

    def route_question(self, state: AgentState) -> AgentState:
        # st.info("Soru yönlendirme aracı çağrıldı...") # Removed for pop-up reduction
        messages = state["messages"]
        last_message = messages[-1]
        
        next_node_decision = ""

        if "hava" in last_message.content.lower() or "sıcaklık" in last_message.content.lower():
            # st.success("Hava durumu rotasına yönlendiriliyor.") # Removed for pop-up reduction
            next_node_decision = "weather"
        elif "ilginç bilgi" in last_message.content.lower() or "bilgi ver" in last_message.content.lower():
            # st.success("İlginç bilgi rotasına yönlendiriliyor.") # Removed for pop-up reduction
            next_node_decision = "fun_fact"
        elif "özetle" in last_message.content.lower() or "özet" in last_message.content.lower():
            # st.success("Özetleme rotasına yönlendiriliyor.") # Removed for pop-up reduction
            next_node_decision = "summarize"
        else:
            place_keywords = ["mekan", "restoran", "kafe", "meyhane", "nereye gideyim", "öneri", "yer"]
            if any(keyword in last_message.content.lower() for keyword in place_keywords):
                # st.success("Mekan arama rotasına yönlendiriliyor.") # Removed for pop-up reduction
                next_node_decision = "search_places"
            else:
                # st.success("Genel yanıt rotasına yönlendiriliyor.") # Removed for pop-up reduction
                next_node_decision = "generate_response"
        
        state["next_node"] = next_node_decision
        return state


def create_workflow():
    # st.info("LangGraph iş akışı oluşturuluyor...") # Removed for pop-up reduction
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
        lambda state: state["next_node"],
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
    # st.success("LangGraph iş akışı başarıyla oluşturuldu.") # Removed for pop-up reduction
    return app

st.set_page_config(page_title="The Light Passenger", layout="wide")
st.title("The Light Passenger 📍")


if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, dict):
        if msg["role"] == "user":
            st.session_state.messages[i] = HumanMessage(content=msg["content"])
        elif msg["role"] == "assistant":
            st.session_state.messages[i] = AIMessage(content=msg["content"])


for message in st.session_state.messages:
    display_role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(display_role):
        st.markdown(message.content, unsafe_allow_html=True)


if prompt := st.chat_input("Nasıl yardımcı olabilirim?"):
    # Append the new human message to session state as a HumanMessage object
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Yanıt oluşturuluyor..."):
        app = create_workflow()
        

        if "conversation_thread_id" not in st.session_state:
            st.session_state.conversation_thread_id = str(uuid.uuid4())
            
        
        config = {"configurable": {"thread_id": st.session_state.conversation_thread_id}}

        # response_placeholder = st.empty() # Not strictly necessary if we display content at the end
        latest_ai_message_content = ""

        try:
            # Pass the entire conversation history from session_state.messages to the LangGraph stream
            for s in app.stream({"messages": st.session_state.messages}, config=config):
                for key in s:
                    node_output = s[key]
                    if "messages" in node_output:
                        for msg in reversed(node_output["messages"]):
                            if isinstance(msg, AIMessage) and msg.content:
                                latest_ai_message_content = msg.content
                                break # Found the latest AI message in this node's output, move to next node's output


            # After the stream completes, use the latest_ai_message_content found
            if latest_ai_message_content:
                sanitized_content = sanitize_markdown(latest_ai_message_content)
                # Append the AI message to session state as an AIMessage object
                st.session_state.messages.append(AIMessage(content=sanitized_content))
                # st.success("Asistan yanıtı başarıyla eklendi.") # Removed for pop-up reduction
                
                with st.chat_message("assistant"):
                    st.markdown(sanitized_content, unsafe_allow_html=True)
            else:
                error_msg = "Üzgünüm, bir yanıt üretemedim. LangGraph akışı tamamlandı ancak AI mesajı bulunamadı. Lütfen tekrar deneyin."
                st.session_state.messages.append(AIMessage(content=error_msg)) # Store as AIMessage
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.error("LangGraph akışı AI mesajı üretmeden tamamlandı.") # Keep this error as it's a critical state

        except Exception as e:
            error_message = f"Bir hata oluştu: {e}. Lütfen daha sonra tekrar deneyin."
            st.error(f"Ana döngüde beklenmedik hata: {str(e)}") # Keep this error
            print(f"ERROR in main loop: {str(e)}")
            st.session_state.messages.append(AIMessage(content=error_message)) # Store as AIMessage
            with st.chat_message("assistant"):
                st.markdown(error_message)
                st.exception(e)   # Keep this exception for debugging
    st.rerun() # Rerun the app to show the latest message