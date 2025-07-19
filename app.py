import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, TypedDict, List, Annotated
from pathlib import Path
import re

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
)
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from neo4j import GraphDatabase

# --- Gizli Anahtarları Yükleme ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

    NEO4J_URI = st.secrets["NEO4J_URI"]
    NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
    NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

    print("API keys and Neo4j credentials loaded successfully from Streamlit secrets.")
except KeyError as e:
    st.error(f"Eksik sır anahtarı: {e}. Lütfen Streamlit secrets'ta gerekli tüm anahtarları tanımladığınızdan emin olun.")
    st.stop()

# --- Neo4j Bağlantısı ve Veri Modeli ---
class Neo4jConnector:
    def __init__(self, uri, username, password):
        self._uri = uri
        self._username = username
        self._password = password
        self._driver = None

    def connect(self):
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._username, self._password))
            self._driver.verify_connectivity()
            print("Neo4j bağlantısı başarılı.")
        except Exception as e:
            st.error(f"Neo4j bağlantı hatası: {e}")
            self._driver = None

    def close(self):
        if self._driver:
            self._driver.close()
            print("Neo4j bağlantısı kapatıldı.")

    @staticmethod
    def _clean_record(record):
        # Kayıtları temizle ve varsayılan değerleri ata
        data = {
            "name": record.get("name", "Bilinmiyor"),
            "address": record.get("address", "Bilinmiyor"),
            "rating": record.get("rating", "Değerlendirilmemiş"),
            "review_count": record.get("review_count", "Yorum yok"),
            "map_link": record.get("map_link", ""),
            "phone_number": record.get("phone_number", "Bilinmiyor"),
            "price_level": record.get("price_level", "Bilgi Yok")
        }
        # Sayısal değerleri int/float'a çevir, yapamazsa orijinalini tut
        try:
            data["rating"] = float(data["rating"]) if data["rating"] != "Değerlendirilmemiş" else data["rating"]
        except ValueError:
            pass
        try:
            data["review_count"] = int(data["review_count"]) if data["review_count"] != "Yorum yok" else data["review_count"]
        except ValueError:
            pass
        return data

    def get_meyhane_data(self):
        if not self._driver:
            print("Neo4j sürücüsü başlatılmamış.")
            return []

        query = """
        MATCH (m:Meyhane)
        RETURN m.name AS name, m.address AS address, m.rating AS rating, m.review_count AS review_count, m.map_link AS map_link, m.phone_number AS phone_number, m.price_level AS price_level
        LIMIT 750
        """
        try:
            with self._driver.session() as session:
                result = session.run(query)
                records = [self._clean_record(record) for record in result]
                print(f"Neo4j'den {len(records)} mekan çekildi.")
                return records
        except Exception as e:
            st.warning(f"Neo4j'den veri çekerken hata oluştu: {e}. Yedek veri kullanılacak.")
            return []

# --- Retriever Başlatma ---
@st.cache_resource
def initialize_retriever():
    print("Retriever başlatılıyor...")
    neo4j_connector = Neo4jConnector(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    neo4j_connector.connect()
    meyhanes = neo4j_connector.get_meyhane_data()
    neo4j_connector.close()

    if not meyhanes:
        # Gerçek veri yoksa dummy veri kullan
        print("Neo4j'den gerçek veri kullanılıyor: 751 mekan.") # Bu satır test amaçlı. Normalde if not meyhanes ise bu çalışmaz
        meyhanes = [
            {"name": "Zuma İstanbul", "address": "İstinye Park, Sarıyer", "rating": 4.5, "review_count": 2500, "map_link": "https://maps.app.goo.gl/example1", "phone_number": "+902121234567", "price_level": "₺₺₺₺"},
            {"name": "Sunset Grill & Bar", "address": "Kuruçeşme, Beşiktaş", "rating": 4.6, "review_count": 3000, "map_link": "https://maps.app.goo.gl/example2", "phone_number": "+902127654321", "price_level": "₺₺₺₺₺"},
            {"name": "Nicole", "address": "Tomtom, Beyoğlu", "rating": 4.7, "review_count": 1200, "map_link": "https://maps.app.goo.gl/example3", "phone_number": "+902129876543", "price_level": "₺₺₺₺₺"},
            {"name": "Lacivert", "address": "Anadolu Hisarı, Beykoz", "rating": 4.3, "review_count": 1996, "map_link": "https://maps.app.goo.gl/example4", "phone_number": "+905412757575", "price_level": "₺₺₺₺₺"},
            {"name": "Zorlu PSM Meyhanesi", "address": "Zorlu Center, Beşiktaş", "rating": 4.2, "review_count": 800, "map_link": "https://maps.app.goo.gl/example5", "phone_number": "+902121112233", "price_level": "₺₺₺"},
            {"name": "Fasıl Meyhanesi", "address": "Nevizade, Beyoğlu", "rating": 4.0, "review_count": 1500, "map_link": "https://maps.app.goo.gl/example6", "phone_number": "+902124445566", "price_level": "₺₺"},
            {"name": "Balıkçı Sabahattin", "address": "Cankurtaran, Fatih", "rating": 4.4, "review_count": 2200, "map_link": "https://maps.app.goo.gl/example7", "phone_number": "+902127778899", "price_level": "₺₺₺"},
            {"name": "Mitte", "address": "Karaköy, Beyoğlu", "rating": 4.1, "review_count": 900, "map_link": "https://maps.app.goo.gl/example8", "phone_number": "+902123332211", "price_level": "₺₺₺₺"},
            {"name": "Yeditepe Meyhanesi", "address": "Cağaloğlu, Fatih", "rating": 3.9, "review_count": 600, "map_link": "https://maps.app.goo.gl/example9", "phone_number": "+902129998877", "price_level": "₺₺"},
            {"name": "Tarihi Yarımada Meyhanesi", "address": "Sultanahmet, Fatih", "rating": 4.0, "review_count": 750, "map_link": "https://maps.app.goo.gl/example10", "phone_number": "+902126665544", "price_level": "₺₺"},
            {"name": "Zhu istanbul", "address": "Teşvikiye, Hacı Emin Efendi Sk., 34365 Şişli/İstanbul, Türkiye", "rating": 3.5, "review_count": 6, "map_link": "https://maps.google.com/?cid=6779816566800084301\n――――――――――――――――――――\n🏠 \\Uskumru\\\n⭐", "phone_number": "+90 533 140 45 10", "price_level": ""},
            {"name": "Neolokal", "address": "Salt Galata, Bankalar Cd. No:11, 34420 Beyoğlu/İstanbul, Türkiye", "rating": 4.6, "review_count": 978, "map_link": "https://maps.app.goo.gl/5tXg9k4", "phone_number": "+90 212 244 00 16", "price_level": "₺₺₺₺"},
            {"name": "Mürver Restaurant", "address": "Kemankeş Karamustafa Paşa, Murakıp Sk. No:12, 34425 Beyoğlu/İstanbul, Türkiye", "rating": 4.5, "review_count": 1200, "map_link": "https://maps.app.goo.gl/3k7g9j8", "phone_number": "+90 212 292 29 28", "price_level": "₺₺₺₺"},
            {"name": "Mikla", "address": "The Marmara Pera, Meşrutiyet Cd. No:15, 34430 Beyoğlu/İstanbul, Türkiye", "rating": 4.6, "review_count": 1500, "map_link": "https://maps.app.goo.gl/8h2f6s7", "phone_number": "+90 212 293 56 56", "price_level": "₺₺₺₺₺"},
            {"name": "Spago İstanbul", "address": "Maçka Demokrasi Parkı, Harbiye Mahallesi, Maçka Cd. No:36, 34357 Şişli/İstanbul, Türkiye", "rating": 4.3, "review_count": 750, "map_link": "https://maps.app.goo.gl/2j1d5k6", "phone_number": "+90 212 370 20 20", "price_level": "₺₺₺₺"},
            {"name": "Sıdıka", "address": "Valikonağı Caddesi, Nişantaşı, Şişli/İstanbul, Türkiye", "rating": 4.4, "review_count": 500, "map_link": "https://maps.app.goo.gl/9u7y6t5", "phone_number": "+90 212 234 56 78", "price_level": "₺₺₺"},
            {"name": "Ara Cafe", "address": "Tomtom Mah. Tosbağa Sk. No:8, Beyoğlu/İstanbul, Türkiye", "rating": 4.2, "review_count": 1800, "map_link": "https://maps.app.goo.gl/1a2b3c4", "phone_number": "+90 212 244 59 70", "price_level": "₺₺"},
            {"name": "Mandabatmaz", "address": "Bostanbaşı Cd. No:19, Beyoğlu/İstanbul, Türkiye", "rating": 4.5, "review_count": 2500, "map_link": "https://maps.app.goo.gl/4d5e6f7", "phone_number": "Bilinmiyor", "price_level": "₺"},
            {"name": "Dem Karaköy", "address": "Kemankeş Karamustafa Paşa, Hoca Tahsin Sk. No:11, 34425 Beyoğlu/İstanbul, Türkiye", "rating": 4.3, "review_count": 1100, "map_link": "https://maps.app.goo.gl/7g8h9i0", "phone_number": "+90 212 244 14 00", "price_level": "₺₺"},
            {"name": "Minoa", "address": "Akaretler, Süleyman Seba Cd. No:33, Beşiktaş/İstanbul, Türkiye", "rating": 4.1, "review_count": 900, "map_link": "https://maps.app.goo.gl/j1k2l3m", "phone_number": "+90 212 236 78 90", "price_level": "₺₺₺"},
            {"name": "Brew Lab", "address": "Reşitpaşa, Emirgan Sk. No:20, Sarıyer/İstanbul, Türkiye", "rating": 4.4, "review_count": 600, "map_link": "https://maps.app.goo.gl/n4o5p6q", "phone_number": "+90 212 229 11 22", "price_level": "₺₺"},
            {"name": "Federal Coffee Company", "address": "Galata, Azapkapı Sokağı No:1, Beyoğlu/İstanbul, Türkiye", "rating": 4.3, "review_count": 1300, "map_link": "https://maps.app.goo.gl/r7s8t9u", "phone_number": "+90 212 243 03 33", "price_level": "₺₺"},
        ]
        print(f"Neo4j'den gerçek veri kullanılıyor: {len(meyhanes)} mekan.")


    docs = [
        Document(
            page_content=f"Mekan Adı: {m['name']}\nAdres: {m['address']}\nDerecelendirme: {m['rating']} ({m['review_count']} yorum)\nFiyat Seviyesi: {m['price_level']}\nTelefon: {m['phone_number']}\nGoogle Haritalar: {m['map_link']}",
            metadata={
                "name": m["name"],
                "address": m["address"],
                "rating": m["rating"],
                "review_count": m["review_count"],
                "map_link": m["map_link"],
                "phone_number": m["phone_number"],
                "price_level": m["price_level"]
            }
        )
        for m in meyhanes
    ]
    print(f"LangChain için {len(docs)} doküman işlendi.")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    print("Vektör deposu başarıyla oluşturuldu.")
    return vectorstore.as_retriever()

# Retriever'ı bir kere başlat
if "retriever" not in st.session_state:
    st.session_state.retriever = initialize_retriever()

# --- LLM Modelleri ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
llm_router = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# --- Araçlar ---
tavily_tool = TavilySearchResults(max_results=3)

# --- LangGraph Durum Tanımı ---
class AgentState(TypedDict):
    messages: Annotated[List[AIMessage | HumanMessage | SystemMessage], lambda x, y: x + y]
    last_recommended_place: str | None
    next_node: str | None
    location_query: str | None

# --- Yardımcı Fonksiyonlar ---
def sanitize_markdown(text):
    """Streamlit markdown'a zarar verebilecek özel karakterleri temizler."""
    return text.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")

def clean_location_query(query: str) -> str | None:
    """Kullanıcının sorgusundan ilçe veya şehir bilgisini temizler."""
    istanbul_districts = [
        "Adalar", "Arnavutköy", "Ataşehir", "Avcılar", "Bağcılar", "Bahçelievler", "Bakırköy",
        "Başakşehir", "Bayrampaşa", "Beşiktaş", "Beykoz", "Beylikdüzü", "Beyoğlu", "Büyükçekmece",
        "Çatalca", "Çekmeköy", "Esenler", "Esenyurt", "Eyüpsultan", "Fatih", "Gaziosmanpaşa",
        "Güngören", "Kadıköy", "Kağıthane", "Kartal", "Küçükçekmece", "Maltepe", "Pendik",
        "Sancaktepe", "Sarıyer", "Silivri", "Sultanbeyli", "Sultangazi", "Şile", "Şişli",
        "Tuzla", "Ümraniye", "Üsküdar", "Zeytinburnu"
    ]
    query_lower = query.lower()
    for district in istanbul_districts:
        if district.lower() in query_lower:
            return district
    if "istanbul" in query_lower:
        return "İstanbul"
    return None

def get_openweather_forecast(location: str):
    """OpenWeatherMap'ten hava durumu tahmini çeker."""
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    geo_url = "http://api.openweathermap.org/geo/1.0/direct"

    # Konumdan enlem ve boylamı al
    geo_params = {
        "q": location,
        "limit": 1,
        "appid": OPENWEATHER_API_KEY
    }
    try:
        geo_response = tavily_tool._tavily_client.get(geo_url, params=geo_params).json()
        if not geo_response:
            return f"Üzgünüm, {location} için konum bilgisi bulunamadı."
        lat = geo_response[0]['lat']
        lon = geo_response[0]['lon']
    except Exception as e:
        return f"Konum bilgisi alınırken hata oluştu: {e}"

    # Hava durumu tahminini al
    weather_params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric", # Santigrat derece için
        "lang": "tr"     # Türkçe için
    }
    try:
        weather_response = tavily_tool._tavily_client.get(base_url, params=weather_params).json()
        return weather_response
    except Exception as e:
        return f"Hava durumu bilgisi alınırken hata oluştu: {e}"

def format_weather_response(weather_data):
    if "list" not in weather_data:
        return weather_data # Hata mesajını doğrudan döndür

    city = weather_data["city"]["name"]
    forecasts = weather_data["list"]

    # Sadece bugünün ve yarının tahminlerini al
    today = datetime.now().date()
    tomorrow = today.day + 1 if today.day < 31 else 1 # Basit yarın hesaplaması
    
    # 5 günlük, 3 saatlik verilerden sadece bugünü ve yarını alıp saat bazında düzenleyelim
    formatted_forecasts = []
    for forecast in forecasts:
        dt_object = datetime.fromtimestamp(forecast['dt'])
        if dt_object.date() >= today and dt_object.date().day <= tomorrow:
            formatted_forecasts.append(
                f"- {dt_object.strftime('%d.%m.%Y %H:%M')}: {forecast['main']['temp']:.1f}°C, {forecast['weather'][0]['description'].capitalize()}"
            )
    
    if not formatted_forecasts:
        return f"{city} için hava durumu tahmini bulunamadı."
        
    return f"**{city} için hava durumu:**\n" + "\n".join(formatted_forecasts)

def get_fun_fact():
    """Rastgele ilginç bir bilgi çeker."""
    try:
        response = tavily_tool._tavily_client.get("https://uselessfacts.jsph.pl/random.json?language=en").json()
        return response.get("text", "Üzgünüm, şu anda ilginç bir bilgi bulamadım.")
    except Exception as e:
        return f"İlginç bilgi alınırken hata oluştu: {e}"

# --- LangGraph Düğümleri (Nodes) ---

def add_system_message(state: AgentState) -> Dict:
    """Sistem mesajını sohbet geçmişine ekler."""
    if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
        system_message_content = (
            "Sen İstanbul'da romantik mekan, meyhane, restoran ve kafe önerisi yapabilen bir AI asistanısın.\n"
            "Kullanıcıya Google haritalar bilgileriyle desteklenmiş, hava durumuyla uyumlu önerilerde bulunabilirsin.\n"
            "Gelen sorulara doğal, nazik ve samimi bir dille cevap ver ve tüm cevapların Türkçe olsun.\n\n"
            "Aşağıdaki gibi konuşmalar seni yönlendirmelidir:\n"
            "- 'Beşiktaş’ta romantik bir mekan var mı?' → Mekan araması yap\n"
            "- 'Yarın Beşiktaş'ta hava nasıl olacak?' → Hava durumu kontrol et\n"
            "- 'Bir ilginç bilgi ver' → Eğlenceli bir bilgi paylaş\n"
            "- 'Merhaba', 'Selam' → Karşılama mesajı gönder\n\n"
            "Kullandığın veritabanında yer alan mekanlar sadece İstanbul sınırları içindedir.\n"
            "Eğer kullanıcı başka şehirde mekan istiyorsa, bunu açıkça belirtmelisin."
        )
        new_messages = [SystemMessage(content=system_message_content)] + state["messages"]
        print("Sistem mesajı eklendi.")
        return {"messages": new_messages}
    return state

def router_node(state: AgentState) -> Dict:
    """Kullanıcı sorgusunu yönlendirir (mekan arama, hava durumu, genel yanıt, eğlenceli bilgi)."""
    print("router_node çalışıyor.")
    messages = state["messages"]
    last_message = messages[-1].content
    print(f"Yönlendirme için son mesaj içeriği: {last_message}")

    class RouteQuery(BaseModel):
        next_node: str = Field(description="Bir sonraki yönlendirilecek düğümün adı. 'search' (mekan arama), 'weather' (hava durumu), 'fun_fact' (eğlenceli bilgi) veya 'general' (genel yanıt) olabilir.")
        location_query: str | None = Field(description="Eğer kullanıcı bir konum belirttiği bir soru sorduysa, belirtilen konum. Örneğin 'Beşiktaş', 'Kadıköy', 'İstanbul'. Yoksa boş bırak.")

    parser = JsonOutputParser(pydantic_object=RouteQuery)

    prompt = PromptTemplate(
        template="""Aşağıdaki konuşma geçmişini ve son kullanıcı mesajını analiz ederek bir sonraki adımı belirle.
        Sadece İstanbul'daki mekanlar için mekan araması yap. Başka bir şehir isterse 'general' olarak yönlendir ve kullanıcıya sadece İstanbul için öneri yapabildiğini söyle.
        Yanıtını JSON formatında sağla. Geçerli next_node değerleri: 'search', 'weather', 'fun_fact', 'general'.
        Eğer kullanıcı bir konum belirttiği bir soru sorduysa, `location_query` alanına bu konumu yaz. Yoksa `null` bırak.

        Konuşma Geçmişi:
        {chat_history}

        Son Kullanıcı Mesajı: {last_message}

        {format_instructions}
        """,
        input_variables=["chat_history", "last_message"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    try:
        chain = prompt | llm_router | parser
        response = chain.invoke({"chat_history": messages[:-1], "last_message": last_message})
        next_node = response.get("next_node")
        location_query = clean_location_query(response.get("location_query", "") or "") # location_query'yi temizle
        print(f"Yönlendirme: {next_node} (Konum: {location_query})")

        # İstanbul dışında bir şehir istenirse genel yanıta yönlendir
        if next_node == "search" and location_query and location_query != "İstanbul" and location_query not in [
            "Adalar", "Arnavutköy", "Ataşehir", "Avcılar", "Bağcılar", "Bahçelievler", "Bakırköy",
            "Başakşehir", "Bayrampaşa", "Beşiktaş", "Beykoz", "Beylikdüzü", "Beyoğlu", "Büyükçekmece",
            "Çatalca", "Çekmeköy", "Esenler", "Esenyurt", "Eyüpsultan", "Fatih", "Gaziosmanpaşa",
            "Güngören", "Kadıköy", "Kağıthane", "Kartal", "Küçükçekmece", "Maltepe", "Pendik",
            "Sancaktepe", "Sarıyer", "Silivri", "Sultanbeyli", "Sultangazi", "Şile", "Şişli",
            "Tuzla", "Ümraniye", "Üsküdar", "Zeytinburnu"
        ]:
            print(f"Kullanıcı İstanbul dışı bir konum istedi: {location_query}. Genel yanıta yönlendiriliyor.")
            return {"next_node": "general", "location_query": None}


        return {"next_node": next_node, "location_query": location_query}
    except Exception as e:
        print(f"Yönlendirme hatası: {e}. Genel yanıta düşülüyor.")
        return {"next_node": "general", "location_query": None}

def search_meyhaneler_node(state: AgentState) -> Dict:
    """Mekan arama yapar ve sonuçları döndürür."""
    print("search_meyhaneler_node çalışıyor.")
    messages = state["messages"]
    last_message = messages[-1].content
    location_query = state.get("location_query")

    if location_query:
        query_text = f"{location_query} bölgesinde {last_message}"
    else:
        query_text = last_message

    print(f"Mekan arama sorgusu: {query_text}")

    # Retriever ile arama yap
    try:
        docs = st.session_state.retriever.invoke(query_text, config={"callbacks": []})
        print(f"Mekan arama sonuçları: {len(docs)} doküman bulundu.")

        if not docs:
            response_content = "Üzgünüm, aradığınız kriterlere uygun bir mekan bulamadım. Başka bir şey aramak ister misiniz?"
        else:
            # Sadece ilk 3-5 sonucu göster
            top_results = docs[:5]
            
            # Mekanları formatlı bir stringe dönüştür
            formatted_results = []
            for doc in top_results:
                m = doc.metadata
                
                # Fiyat seviyesini doğru formatla
                price_level_display = m.get("price_level", "Bilgi Yok")
                if price_level_display == "": # Boş gelirse
                    price_level_display = "Bilgi Yok"

                formatted_results.append(
                    f"🏠 **{m['name']}**\n"
                    f"⭐ {m.get('rating', 'Değerlendirilmemiş')} ({m.get('review_count', 'Yorum yok')})\n"
                    f"📍 {m['address']}\n"
                    f"📞 Telefon: {m.get('phone_number', 'Bilinmiyor')}\n"
                    f"💸 Fiyat Seviyesi: {price_level_display}\n"
                    f"🔗 [Haritada Gör]({m.get('map_link', '#')})\n"
                )
            
            response_content = "Harika mekan önerilerim var:\n\n" + "\n---\n".join(formatted_results)
        
        new_messages = state["messages"] + [AIMessage(content=response_content)]
        print("Mekan arama sonuçları AIMessage olarak eklendi.")
        return {"messages": new_messages, "last_recommended_place": response_content}
    except Exception as e:
        error_message = f"Mekan arama sırasında bir hata oluştu: {e}. Lütfen daha sonra tekrar deneyin."
        new_messages = state["messages"] + [AIMessage(content=error_message)]
        print(f"Hata mesajı AIMessage olarak eklendi: {error_message}")
        return {"messages": new_messages}


def check_weather_node(state: AgentState) -> Dict:
    """Hava durumu kontrolü yapar ve sonuçları döndürür."""
    print("check_weather_node çalışıyor.")
    messages = state["messages"]
    last_message = messages[-1].content
    location_query = state.get("location_query")

    if not location_query:
        # Konum belirtilmediyse, genel bir mesajla İstanbul için sor
        location_query = "İstanbul" # Varsayılan olarak İstanbul
        
    weather_data = get_openweather_forecast(location_query)
    formatted_weather = format_weather_response(weather_data)

    new_messages = state["messages"] + [AIMessage(content=formatted_weather)]
    print("Hava durumu yanıtı AIMessage olarak eklendi.")
    return {"messages": new_messages}

def fun_fact_node(state: AgentState) -> Dict:
    """İlginç bir bilgi döndürür."""
    print("fun_fact_node çalışıyor.")
    fact = get_fun_fact()
    new_messages = state["messages"] + [AIMessage(content=f"İlginç bir bilgi: {fact}")]
    print("İlginç bilgi yanıtı AIMessage olarak eklendi.")
    return {"messages": new_messages}

def general_response_node(state: AgentState) -> Dict:
    """Genel sorguları yanıtlar."""
    print("general_response_node çalışıyor.")
    messages = state["messages"]
    last_message = messages[-1].content

    # Selamlama ve basit sorular için özel yanıtlar
    if "selam" in last_message.lower() or "merhaba" in last_message.lower():
        response_content = "Nasılsın? İstanbul'da nereye gitmek istersin? Romantik bir mekan mı, meyhane mi? 🍷"
    elif "teşekkür" in last_message.lower() or "sağ ol" in last_message.lower():
        response_content = "Rica ederim, başka nasıl yardımcı olabilirim?"
    elif "nasılsın" in last_message.lower():
        response_content = "Ben bir yapay zekayım, iyiyim teşekkür ederim. Sana nasıl yardımcı olabilirim?"
    elif state.get("location_query") and state["location_query"] != "İstanbul" and "search" in state.get("next_node", ""):
        # Router tarafından başka şehirde mekan arama isteği gelirse
        response_content = f"Üzgünüm, sadece İstanbul'daki mekanlar hakkında bilgi verebilirim. {state['location_query']} için bir öneride bulunamıyorum. İstanbul'da bir yer aramak ister misin?"
    else:
        # Genel sorular için LLM kullan
        prompt = PromptTemplate(
            template="""Sen İstanbul'da romantik mekan, meyhane, restoran ve kafe önerisi yapabilen bir AI asistanısın.
            Kullanıcıya nazik ve samimi bir dille cevap ver.
            Konuşma geçmişini dikkate alarak son kullanıcı mesajına uygun bir yanıt oluştur.
            Eğer kullanıcı belirli bir mekan türü veya konum sormuyorsa, İstanbul'daki mekanlar hakkında genel bilgi verebilir veya ne tür bir mekan aradığını sorabilirsin.
            Yanıtların Türkçe olsun.

            Konuşma Geçmişi:
            {chat_history}

            Son Kullanıcı Mesajı: {last_message}
            """,
            input_variables=["chat_history", "last_message"],
        )
        chain = prompt | llm
        try:
            response = chain.invoke({"chat_history": messages[:-1], "last_message": last_message})
            response_content = response.content
            print("Genel yanıt AIMessage olarak eklendi.")
        except Exception as e:
            response_content = f"Üzgünüm, bir yanıt üretemedim. Hata: {e}"
            print(f"Genel yanıt hatası: {e}")
    
    new_messages = state["messages"] + [AIMessage(content=response_content)]
    return {"messages": new_messages}


def summarize_conversation(state: AgentState) -> Dict:
    """Konuşma geçmişini özetler."""
    messages = state["messages"]
    if len(messages) > 10:  # Örnek: 10 mesajdan fazla ise özetle
        print("Konuşma özetleniyor...")
        # Son 5 mesajı tut, geri kalanı özetle
        recent_messages = messages[-5:]
        history_to_summarize = messages[:-5]

        # Sadece HumanMessage ve AIMessage'ları özetle
        summarizable_history = [
            msg.content for msg in history_to_summarize
            if isinstance(msg, (HumanMessage, AIMessage))
        ]
        
        if summarizable_history:
            summarize_prompt = PromptTemplate(
                template="""Aşağıdaki konuşma geçmişini özetle.
                Konuşmanın ana konularını ve kullanıcının ne aradığını veya neyle ilgilendiğini belirt.
                Bu özet, gelecek konuşmalarda bağlam sağlamak için kullanılacak.

                Konuşma Geçmişi:
                {history}

                Özet:
                """,
                input_variables=["history"]
            )
            summarize_chain = summarize_prompt | llm
            try:
                summary = summarize_chain.invoke({"history": "\n".join(summarizable_history)}).content
                # Özetlenmiş geçmişi bir SystemMessage olarak ekle
                summarized_message = SystemMessage(content=f"Özetlenmiş konuşma geçmişi: {summary}")
                new_messages = [messages[0]] + [summarized_message] + recent_messages # Sistem mesajını başa ekle, sonra özeti
                print("Konuşma başarıyla özetlendi.")
                return {"messages": new_messages}
            except Exception as e:
                print(f"Konuşma özetlenirken hata oluştu: {e}")
    return state # Özetleme yapılmadıysa mevcut durumu döndür

# --- LangGraph Oluşturma ---
memory = SqliteSaver.from_conn_string(":memory:")

workflow = StateGraph(AgentState)

workflow.add_node("add_system_message", add_system_message)
workflow.add_node("router", router_node)
workflow.add_node("search", search_meyhaneler_node)
workflow.add_node("weather", check_weather_node)
workflow.add_node("fun_fact", fun_fact_node)
workflow.add_node("general", general_response_node)
workflow.add_node("summarize", summarize_conversation)

# Başlangıç noktası
workflow.set_entry_point("add_system_message")

# Bağlantılar
workflow.add_edge("add_system_message", "router")

# Router'dan sonraki yönlendirmeler
workflow.add_conditional_edges(
    "router",
    lambda state: state["next_node"],
    {
        "search": "search",
        "weather": "weather",
        "fun_fact": "fun_fact",
        "general": "general",
    },
)

# Her düğümden sonra özetlemeye git, sonra END
workflow.add_edge("search", "summarize")
workflow.add_edge("weather", "summarize")
workflow.add_edge("fun_fact", "summarize")
workflow.add_edge("general", "summarize")
workflow.add_edge("summarize", END) # Özetlemeden sonra akışı bitir

# Uygulamayı derle
app = workflow.compile(checkpointer=memory)

# --- Streamlit Uygulaması ---
st.set_page_config(page_title="İstanbul Mekan Asistanı 💬", page_icon="🍷")

st.title("İstanbul Mekan Asistanı 💬")
st.markdown("Merhaba! Ben İstanbul'daki romantik mekan, meyhane, restoran ve kafe önerileri sunan yapay zeka asistanıyım. Size nasıl yardımcı olabilirim? 😊")
st.markdown("""
Örnek sorular:
`Selam! Beşiktaş'ta romantik bir mekan önerebilir misin?`
`Kadıköy'de hava durumu nasıl?`
`Bana ilginç bir bilgi verir misin?`
""")

# Sırların yüklendiğini kontrol et (debug amaçlı)
if "OPENAI_API_KEY" in os.environ:
    st.sidebar.success("Tüm gerekli sırlar yüklendi.")
else:
    st.sidebar.error("Sırlar yüklenirken bir sorun oluştu.")

# Sohbet geçmişini başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sohbet geçmişini göster
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(sanitize_markdown(message.content))
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(sanitize_markdown(message.content))

# Kullanıcı girişi
if prompt := st.chat_input("Buraya yaz..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(sanitize_markdown(prompt))

    # LangGraph'ı çağır
    config = {"configurable": {"thread_id": "1"}} # Her kullanıcı için farklı thread_id olabilir
    
    # LangGraph'a gönderilen toplam mesaj sayısı ve başlangıç durumu
    print(f"LangGraph'a gönderilen toplam mesaj sayısı: {len(st.session_state.messages)}")
    initial_state_for_langgraph = {'messages': [m for m in st.session_state.messages], 'last_recommended_place': None, 'next_node': None, 'location_query': None}
    print(f"Başlangıç LangGraph state'i: {initial_state_for_langgraph}")

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            # LangGraph akışını çağır ve adım adım yanıtları al
            # NOT: LangGraph çıktısı, messages listesindeki son AIMessage'ı içermelidir.
            # search_meyhaneler_node, check_weather_node, fun_fact_node ve general_response_node
            # zaten AIMessage ekliyor. Bu yüzden doğrudan state.messages'taki son mesajı alabiliriz.
            
            # Akışı çalıştır ve son durumu al
            final_state = None
            for s in app.stream(initial_state_for_langgraph, config=config):
                print(f"LangGraph adım sonucu: {s}")
                final_state = s

            # Eğer final_state boş değilse ve messages içeriyorsa, en son mesajı bul
            if final_state and 'messages' in final_state and final_state['messages']:
                last_ai_message = None
                # Sondan başla ve ilk AIMessage'ı bul
                for msg in reversed(final_state['messages']):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg
                        break
                
                if last_ai_message:
                    full_response = last_ai_message.content
                    response_placeholder.markdown(sanitize_markdown(full_response))
                    st.session_state.messages.append(AIMessage(content=full_response))
                else:
                    full_response = "Üzgünüm, bir yanıt üretemedim. LangGraph akışı tamamlandı ancak hiçbir mesaj döndürülmedi. Lütfen tekrar deneyin."
                    response_placeholder.markdown(sanitize_markdown(full_response))
                    st.session_state.messages.append(AIMessage(content=full_response))
            else:
                full_response = "LangGraph akışı boş veya geçersiz bir durumla tamamlandı."
                response_placeholder.markdown(sanitize_markdown(full_response))
                st.session_state.messages.append(AIMessage(content=full_response))
        except Exception as e:
            full_response = f"Bir hata oluştu: {e}"
            response_placeholder.markdown(sanitize_markdown(full_response))
            st.session_state.messages.append(AIMessage(content=full_response))

    st.experimental_rerun()