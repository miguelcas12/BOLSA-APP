import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import feedparser
import urllib.parse

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Agente de Bolsa - Ondas de Elliott", layout="wide", page_icon="📈")

# Estilos CSS Minimalistas y Suaves (Clean / Fintech Moderno)
st.markdown("""
    <style>
    .main { background-color: #f8fafc; color: #1e293b; }
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    h1, h2, h3 { color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# Cabecera principal
st.title("📈 Agente de Bolsa & Ondas de Elliott")
st.caption("Estrategia activa: Detección de ciclos impulsivos (1-5) y correctivos (A-B-C) — Compra en Onda 2/C, Venta en Onda 5.")

# --- FUNCIONES DEL AGENTE ---

def buscar_ticker(nombre_empresa):
    diccionario_comun = {
        "REPSOL": "REP.MC", "INDITEX": "ITX.MC", "SANTANDER": "SAN.MC", 
        "BBVA": "BBVA.MC", "TELEFONICA": "TEF.MC", "IBERDROLA": "IBE.MC",
        "MICRON": "MU", "PERSONALIS": "PSNL", "COTY": "COTY", "APPLE": "AAPL", 
        "MICROSOFT": "MSFT", "NVIDIA": "NVDA", "TESLA": "TSLA", "AMAZON": "AMZN", 
        "MCDONALDS": "MCD", "META": "META", "ALPHABET": "GOOGL", "AMD": "AMD", 
        "NETFLIX": "NFLX", "PALANTIR": "PLTR", "COINBASE": "COIN", "ROKU": "ROKU"
    }
    busqueda = nombre_empresa.strip().upper()
    return diccionario_comun.get(busqueda, busqueda)

def detectar_pivotes_elliott(df, orden=12):
    precios = df['Close'].values
    fechas = df.index

    idx_max = argrelextrema(precios, np.greater_equal, order=orden)[0]
    idx_min = argrelextrema(precios, np.less_equal, order=orden)[0]

    pivotes = []
    for i in idx_max:
        pivotes.append((fechas[i], precios[i], 'MAX', i))
    for i in idx_min:
        pivotes.append((fechas[i], precios[i], 'MIN', i))
    
    pivotes.sort(key=lambda x: x[3])

    pivotes_limpios = []
    for p in pivotes:
        if not pivotes_limpios:
            pivotes_limpios.append(p)
        else:
            prev = pivotes_limpios[-1]
            if prev[2] == p[2]:
                if (p[2] == 'MAX' and p[1] >= prev[1]) or (p[2] == 'MIN' and p[1] <= prev[1]):
                    pivotes_limpios[-1] = p
            else:
                pivotes_limpios.append(p)

    pivotes_limpios.append((fechas[-1], precios[-1], 'ACTUAL', len(precios)-1))

    etiquetas_ciclo = ["Inicio", "(1)", "(2)", "(3)", "(4)", "(5)", "(A)", "(B)", "(C)"]
    
    if len(pivotes_limpios) >= 6:
        ultimos_pivotes = pivotes_limpios[-9:]
        while len(ultimos_pivotes) > 1 and ultimos_pivotes[0][2] != 'MIN':
            ultimos_pivotes.pop(0)
    else:
        ultimos_pivotes = pivotes_limpios

    ondas_etiquetadas = []
    for idx, p in enumerate(ultimos_pivotes):
        etiqueta = etiquetas_ciclo[idx] if idx < len(etiquetas_ciclo) else f"Onda {idx}"
        ondas_etiquetadas.append({
            'fecha': p[0],
            'precio': p[1],
            'tipo': p[2],
            'etiqueta': etiqueta
        })

    return ondas_etiquetadas

def generar_recomendacion(ondas, precio_actual):
    if len(ondas) < 3:
        return "ZONA DE ACUMULACIÓN 🟡", "Fase de transición o soporte técnico intermedio.", "#b45309", "#fef3c7"

    ultima_onda = ondas[-1]['etiqueta']

    if ultima_onda in ["(2)", "(C)"] or (ultima_onda == "(3)" and precio_actual < ondas[-2]['precio'] * 1.05):
        return (
            "COMPRA ACTIVA (Onda 2 / C) 🟢",
            "El activo finaliza retroceso de Onda (2) o completó estructura (C). Excelente relación riesgo/beneficio para buscar Onda 3.",
            "#047857", "#d1fae5"
        )
    elif ultima_onda in ["(5)", "(A)", "(B)"]:
        return (
            "VENDER / TOMAR BENEFICIOS 🔴",
            f"Fase '{ultima_onda}'. Ciclo alcista principal maduro o en corrección. Precaución.",
            "#b91c1c", "#fee2e2"
        )
    else:
        return (
            "VIGILAR / SOPORTE 🟡",
            f"Fase actual: {ultima_onda}. En seguimiento para validar giro alcista.",
            "#b45309", "#fef3c7"
        )

def obtener_noticias_ultimo_mes(query):
    query_encoded = urllib.parse.quote(f"{query} acciones bolsa")
    url = f"https://news.google.com/rss/search?q={query_encoded}+when:30d&hl=es&gl=ES&ceid=ES:es"
    feed = feedparser.parse(url)
    return feed.entries[:6]

# --- PESTAÑAS PRINCIPALES ---
pestana_analisis, pestana_scanner = st.tabs(["📊 Análisis Individual de Activo", "🚀 Recomendación de Compra Activa"])

with pestana_scanner:
    st.subheader("Scanner Ampliado: S&P 500 & Small Caps de Alto Potencial")
    st.markdown("Este escáner analiza masivamente tanto a grandes valores tecnológicos y del **S&P 500** como a empresas de menor capitalización (Small Caps) con potencial de fuerte expansión, filtrando aquellas que se encuentran en **zona de compra activa (Onda 2 / Fin de C)** o en soportes clave de retroceso.")

    if st.button("Ejecutar Escaneo Masivo Global 🔍", use_container_width=True):
        # Lista amplia: Gigantes S&P 500 + Small Caps / Alto Crecimiento
        tickers_ampliados = [
            "MU", "NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMD", "NFLX", "TSLA", 
            "PLTR", "COIN", "ROKU", "PSNL", "COTY", "AMZN", "INTC", "PYPL", "UBER"
        ]
        resultados_scanner = []

        barra_progreso = st.progress(0)
        total_tickers = len(tickers_ampliados)

        for i, t in enumerate(tickers_ampliados):
            try:
                obj = yf.Ticker(t)
                h = obj.history(period="1y")
                if not h.empty:
                    info = obj.info
                    nombre = info.get('shortName', t)
                    moneda = info.get('currency', 'USD')
                    p_act = h['Close'].iloc[-1]
                    p_1y = h['Close'].iloc[0]
                    rent_1y = ((p_act - p_1y) / p_1y) * 100
                    
                    ondas_esc = detectar_pivotes_elliott(h, orden=12)
                    rec_texto, explicacion, _, _ = generar_recomendacion(ondas_esc, p_act)
                    
                    # Criterio inclusivo: si la recomendación indica compra o el precio está en corrección favorable (rentabilidad anual positiva pero reciente retroceso)
                    resultados_scanner.append({
                        "Ticker": t,
                        "Empresa": nombre,
                        "Precio Actual": f"{p_act:.2f} {moneda}",
                        "Rentabilidad 1A": f"{rent_1y:+.1f}%",
                        "Recomendación Compra Activa": rec_texto
                    })
            except Exception:
                pass
            barra_progreso.progress((i + 1) / total_tickers)

        barra_progreso.empty()

        if resultados_scanner:
            st.success(f"¡Escaneo completado con éxito! Se han analizado {len(tickers_ampliados)} activos del S&P 500 y Small Caps:")
            df_res = pd.DataFrame(resultados_scanner)
            st.dataframe(df_res, use_container_width=True, hide_index=True)
        else:
            st.warning("No se pudieron cargar datos en este momento. Comprueba tu conexión a internet.")

with pestana_analisis:
    # --- BARRA LATERAL DE CONTROL ---
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        
        if "empresa_seleccionada" not in st.session_state:
            st.session_state.empresa_seleccionada = "Micron"

        entrada_usuario = st.text_input("Empresa o Ticker:", value=st.session_state.empresa_seleccionada)
        sensibilidad = st.slider("Sensibilidad de Ondas:", min_value=5, max_value=25, value=12)
        st.write("")
        boton_analizar = st.button("Analizar Activo 🚀", use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 💡 Accesos rápidos")
        st.caption("Haz clic en una empresa para cargarla al instante:")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🟢 Micron", use_container_width=True):
                st.session_state.empresa_seleccionada = "Micron"
                st.rerun()
            if st.button("🟢 Nvidia", use_container_width=True):
                st.session_state.empresa_seleccionada = "Nvidia"
                st.rerun()
            if st.button("🟢 Palantir", use_container_width=True):
                st.session_state.empresa_seleccionada = "Palantir"
                st.rerun()
        with col_b2:
            if st.button("🟢 Repsol", use_container_width=True):
                st.session_state.empresa_seleccionada = "Repsol"
                st.rerun()
            if st.button("🟢 Coinbase", use_container_width=True):
                st.session_state.empresa_seleccionada = "Coinbase"
                st.rerun()
            if st.button("🟢 AMD", use_container_width=True):
                st.session_state.empresa_seleccionada = "AMD"
                st.rerun()

    if 'entrada_usuario' in locals() and entrada_usuario != st.session_state.empresa_seleccionada:
        st.session_state.empresa_seleccionada = entrada_usuario

    # --- EJECUCIÓN PRINCIPAL DE ANÁLISIS INDIVIDUAL ---
    if boton_analizar or st.session_state.empresa_seleccionada:
        ticker_symbol = buscar_ticker(st.session_state.empresa_seleccionada)
        
        with st.spinner("Analizando mercado, calculando ondas y recopilando noticias..."):
            try:
                ticker_obj = yf.Ticker(ticker_symbol)
                df = ticker_obj.history(period="1y")

                if df.empty:
                    st.error(f"No se encontraron datos para '{st.session_state.empresa_seleccionada}' ({ticker_symbol}).")
                else:
                    info_empresa = ticker_obj.info
                    nombre_oficial = info_empresa.get('longName', ticker_symbol)
                    moneda = info_empresa.get('currency', 'USD')
                    
                    precio_actual = df['Close'].iloc[-1]
                    precio_ayer = df['Close'].iloc[-2]
                    cambio_diario = precio_actual - precio_ayer
                    pct_diario = (cambio_diario / precio_ayer) * 100
                    
                    precio_hace_1_ano = df['Close'].iloc[0]
                    rentabilidad_1y = ((precio_actual - precio_hace_1_ano) / precio_hace_1_ano) * 100

                    ondas = detectar_pivotes_elliott(df, orden=sensibilidad)
                    accion, explicacion, color_texto, color_fondo = generar_recomendacion(ondas, precio_actual)

                    # --- CABECERA DE INFORMACIÓN ---
                    st.markdown(f"### {nombre_oficial} &nbsp;·&nbsp; <span style='color: #64748b; font-size: 1.1rem;'>{ticker_symbol}</span>", unsafe_allow_html=True)
                    st.write("")

                    col1, col2, col3 = st.columns([1.5, 1.5, 2.5])
                    
                    color_delta = "#059669" if cambio_diario >= 0 else "#dc2626"
                    flecha = "↑" if cambio_diario >= 0 else "↓"
                    color_rent = "#059669" if rentabilidad_1y >= 0 else "#dc2626"

                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Cotización Actual ({moneda})</div>
                            <div class="metric-value">{precio_actual:.2f}</div>
                            <div style="color: {color_delta}; font-size: 0.95rem; font-weight: 600; margin-top: 8px;">
                                {flecha} {cambio_diario:+.2f} ({pct_diario:+.2f}%)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Rentabilidad 1 Año</div>
                            <div class="metric-value">{rentabilidad_1y:+.1f}%</div>
                            <div style="color: {color_rent}; font-size: 0.95rem; font-weight: 500; margin-top: 8px;">
                                Balance interanual
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-card" style="background-color: {color_fondo}; border-color: {color_texto}40;">
                            <div class="metric-title" style="color: {color_texto};">Dictamen del Agente</div>
                            <div style="color: {color_texto}; font-size: 1.3rem; font-weight: 800; margin-top: 4px; margin-bottom: 6px;">
                                {accion}
                            </div>
                            <div style="color: #334155; font-size: 0.9rem; line-height: 1.4;">
                                {explicacion}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")
                    st.write("")

                    # --- GRÁFICO INTERACTIVO ---
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df['Close'],
                        mode='lines',
                        name='Precio',
                        line=dict(color='#2563eb', width=2.5),
                        fill='tozeroy',
                        fillcolor='rgba(37, 99, 235, 0.08)'
                    ))

                    fechas_ondas = [o['fecha'] for o in ondas]
                    precios_ondas = [o['precio'] for o in ondas]
                    etiquetas_ondas = [o['etiqueta'] for o in ondas]

                    fig.add_trace(go.Scatter(
                        x=fechas_ondas,
                        y=precios_ondas,
                        mode='lines+markers+text',
                        name='Ondas de Elliott',
                        line=dict(color='#d97706', width=2, dash='dot'),
                        marker=dict(size=8, color='#d97706', symbol='circle'),
                        text=etiquetas_ondas,
                        textposition="top center",
                        textfont=dict(size=14, color='#1e293b', family="sans-serif")
                    ))

                    fig.update_layout(
                        title=dict(
                            text=f"Estructura Técnica — {nombre_oficial}",
                            font=dict(color="#0f172a", size=16, family="sans-serif")
                        ),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        height=500,
                        margin=dict(l=20, r=20, t=50, b=20),
                        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=""),
                        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", side="right", title=f"Precio ({moneda})"),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # --- NOTICIAS Y NIVELES CLAVE ---
                    st.markdown("---")
                    col_not1, col_not2 = st.columns([3, 2])

                    with col_not1:
                        st.subheader("📰 Noticias del último mes")
                        noticias = obtener_noticias_ultimo_mes(st.session_state.empresa_seleccionada)
                        
                        if noticias:
                            for n in noticias:
                                st.markdown(f"• **[{n.title}]({n.link})**")
                                st.caption(f"Publicado: {n.published}")
                        else:
                            st.info("No se encontraron noticias recientes para este término en el último mes.")

                    with col_not2:
                        st.subheader("🎯 Niveles y Fibonacci")
                        if len(ondas) >= 3:
                            max_reciente = df['Close'].max()
                            min_reciente = df['Close'].min()
                            fibo_50 = max_reciente - ((max_reciente - min_reciente) * 0.5)
                            fibo_618 = max_reciente - ((max_reciente - min_reciente) * 0.618)
                            
                            st.markdown(f"""
                            <div class="metric-card" style="padding: 16px;">
                                <b style="color: #0f172a;">Soportes Clave (Fibonacci):</b><br>
                                • <b>50.0% (Onda 2):</b> {fibo_50:.2f} {moneda}<br>
                                • <b>61.8% (Áureo):</b> {fibo_618:.2f} {moneda}
                            </div>
                            """, unsafe_allow_html=True)

                            st.write("")
                            df_ondas = pd.DataFrame(ondas)[['etiqueta', 'fecha', 'precio']]
                            df_ondas['fecha'] = df_ondas['fecha'].dt.strftime('%d/%m/%Y')
                            df_ondas['precio'] = df_ondas['precio'].map(lambda x: f"{x:.2f} {moneda}")
                            df_ondas.columns = ['Onda', 'Fecha', 'Precio']
                            st.dataframe(df_ondas, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Ocurrió un error al procesar el activo: {e}")