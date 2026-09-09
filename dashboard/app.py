from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


# CONFIGURACIÓN GENERAL

st.set_page_config(
    page_title="Battery Materials Screening",
    page_icon="⚡",
    layout="wide",
)


COLORES_METRICAS = {
    "MAE": "#A9CBE8",   # azul pastel
    "RMSE": "#B85C73",  # granate
    "R2": "#F6E7B0",    # vainilla
}



st.title("⚡ Battery Materials Screening")

st.markdown(
    """
    Dashboard interactivo para explorar los resultados del screening
    de materiales candidatos para electrodos de baterías.

    El pipeline combina:

    - XGBoost para la predicción de `average_voltage`
    - GINE para la predicción de `max_delta_volume`
    - filtros de estabilidad
    - ranking final de candidatos
    """
)

st.divider()



# RUTAS


BASE_DIR = Path(__file__).resolve().parent.parent

SCREENING_PATH = (
    BASE_DIR
    / "resultados_screening"
    / "screening_predicciones.csv"
)

CANDIDATOS_PATH = (
    BASE_DIR
    / "resultados_screening"
    / "candidatos_finales.csv"
)


# CARGA DE DATOS

@st.cache_data
def cargar_datos():

    screening = pd.read_csv(
        SCREENING_PATH
    )

    candidatos = pd.read_csv(
        CANDIDATOS_PATH
    )

    return screening, candidatos


if not SCREENING_PATH.exists():

    st.error(
        f"No se encuentra el archivo:\n\n{SCREENING_PATH}"
    )

    st.stop()


if not CANDIDATOS_PATH.exists():

    st.error(
        f"No se encuentra el archivo:\n\n{CANDIDATOS_PATH}"
    )

    st.stop()


screening, candidatos = cargar_datos()



# CONVERSIÓN DE COLUMNAS NUMÉRICAS

columnas_numericas = [
    "pred_average_voltage",
    "pred_max_delta_volume",
    "energy_above_hull",
]


for df_temp in [screening, candidatos]:

    for columna in columnas_numericas:

        if columna in df_temp.columns:

            df_temp[columna] = pd.to_numeric(
                df_temp[columna],
                errors="coerce",
            )



# SIDEBAR

st.sidebar.header("⚙️ Filtros de screening")

df = candidatos.copy()



# FILTRO WORKING ION

if "working_ion" in df.columns:

    iones = sorted(
        df["working_ion"]
        .dropna()
        .astype(str)
        .unique()
    )

    iones_seleccionados = st.sidebar.multiselect(
        "Working ion",
        options=iones,
        default=iones,
    )

    df = df[
        df["working_ion"]
        .astype(str)
        .isin(iones_seleccionados)
    ]


# FILTRO VOLTAJE

if "pred_average_voltage" in candidatos.columns:

    valores_voltage = candidatos[
        "pred_average_voltage"
    ].dropna()

    if not valores_voltage.empty:

        voltage_min = float(
            valores_voltage.min()
        )

        voltage_max = float(
            valores_voltage.max()
        )

        if voltage_min < voltage_max:

            rango_voltage = st.sidebar.slider(
                "Voltaje predicho (V)",
                min_value=voltage_min,
                max_value=voltage_max,
                value=(
                    voltage_min,
                    voltage_max,
                ),
                step=0.05,
            )

            df = df[
                df["pred_average_voltage"].between(
                    rango_voltage[0],
                    rango_voltage[1],
                )
            ]

        else:

            st.sidebar.info(
                f"Todos los candidatos tienen el mismo "
                f"voltaje: {voltage_min:.3f} V"
            )


# FILTRO CAMBIO DE VOLUMEN

if "pred_max_delta_volume" in candidatos.columns:

    valores_volume = candidatos[
        "pred_max_delta_volume"
    ].dropna()

    if not valores_volume.empty:

        volume_min = float(
            valores_volume.min()
        )

        volume_max = float(
            valores_volume.max()
        )

        if volume_min < volume_max:

            rango_volume = st.sidebar.slider(
                "Cambio de volumen predicho",
                min_value=volume_min,
                max_value=volume_max,
                value=(
                    volume_min,
                    volume_max,
                ),
                step=0.01,
            )

            df = df[
                df["pred_max_delta_volume"].between(
                    rango_volume[0],
                    rango_volume[1],
                )
            ]

        else:

            st.sidebar.info(
                f"Todos los candidatos tienen el mismo "
                f"cambio de volumen predicho: {volume_min:.4f}"
            )



# FILTRO ENERGY ABOVE HULL

if "energy_above_hull" in candidatos.columns:

    valores_ehull = candidatos[
        "energy_above_hull"
    ].dropna()

    if not valores_ehull.empty:

        ehull_min = float(
            valores_ehull.min()
        )

        ehull_max = float(
            valores_ehull.max()
        )

        if ehull_min < ehull_max:

            ehull_limite = st.sidebar.slider(
                "Energy above hull máxima (eV/atom)",
                min_value=ehull_min,
                max_value=ehull_max,
                value=ehull_max,
                step=0.005,
            )

            df = df[
                df["energy_above_hull"]
                <= ehull_limite
            ]

        else:

            st.sidebar.info(
                f"Todos los candidatos tienen "
                f"E above hull = {ehull_min:.4f} eV/atom"
            )



# PESTAÑAS

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔎 Screening",
        "🏆 Candidatos",
        "📊 Exploración",
        "🧠 Modelos",
    ]
)


# 1 - SCREENING


with tab1:

    st.header(
        "Screening de materiales"
    )

    # KPIs
    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Materiales evaluados",
        f"{len(screening):,}",
    )

    col2.metric(
        "Shortlist final",
        f"{len(candidatos):,}",
    )

    col3.metric(
        "Tras aplicar filtros",
        f"{len(df):,}",
    )

    if (
        len(df) > 0
        and "pred_average_voltage" in df.columns
    ):

        col4.metric(
            "Voltaje medio",
            f"{df['pred_average_voltage'].mean():.2f} V",
        )

    else:

        col4.metric(
            "Voltaje medio",
            "—",
        )


    st.divider()


    # GRÁFICO VOLTAJE VS VOLUMEN

    st.subheader(
        "⚡ Trade-off entre voltaje y cambio de volumen"
    )


    if (
        len(df) > 0
        and "pred_average_voltage" in df.columns
        and "pred_max_delta_volume" in df.columns
    ):

        hover_data = {}

        if "material_id" in df.columns:

            hover_data[
                "material_id"
            ] = True


        if "framework_formula" in df.columns:

            hover_data[
                "framework_formula"
            ] = True


        if "energy_above_hull" in df.columns:

            hover_data[
                "energy_above_hull"
            ] = ":.4f"


        if "is_stable" in df.columns:

            hover_data[
                "is_stable"
            ] = True


        hover_data[
            "pred_average_voltage"
        ] = ":.3f"

        hover_data[
            "pred_max_delta_volume"
        ] = ":.4f"


        scatter_kwargs = {
            "data_frame": df,
            "x": "pred_max_delta_volume",
            "y": "pred_average_voltage",
            "hover_data": hover_data,
            "labels": {
                "pred_max_delta_volume":
                    "Cambio de volumen predicho",
                "pred_average_voltage":
                    "Voltaje medio predicho (V)",
                "working_ion":
                    "Working ion",
            },
            "title":
                "Voltaje predicho vs cambio de volumen predicho",
        }


        if "working_ion" in df.columns:

            scatter_kwargs[
                "color"
            ] = "working_ion"


        if "formula_pretty" in df.columns:

            scatter_kwargs[
                "hover_name"
            ] = "formula_pretty"


        fig = px.scatter(
            **scatter_kwargs
        )


        fig.update_layout(
            height=600
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        if (
            df["pred_max_delta_volume"]
            .nunique(dropna=True)
            == 1
        ):

            st.warning(
                "Todos los candidatos presentan prácticamente "
                "el mismo cambio de volumen predicho. "
                "Esto refleja la concentración observada "
                "en las predicciones del modelo GNN."
            )

        else:

            st.info(
                "Los materiales situados hacia la zona "
                "superior izquierda combinan mayor voltaje "
                "con menor cambio de volumen."
            )


    else:

        st.warning(
            "No hay datos suficientes para construir "
            "el gráfico."
        )



#  2 - CANDIDATOS

with tab2:

    st.header(
        "🏆 Candidatos seleccionados"
    )


    if len(df) == 0:

        st.warning(
            "No hay candidatos que cumplan "
            "los filtros actuales."
        )


    else:

        ranking = df.copy()


        columnas_orden = []
        ascending = []


        if "pred_average_voltage" in ranking.columns:

            columnas_orden.append(
                "pred_average_voltage"
            )

            ascending.append(
                False
            )


        if "pred_max_delta_volume" in ranking.columns:

            columnas_orden.append(
                "pred_max_delta_volume"
            )

            ascending.append(
                True
            )


        if columnas_orden:

            ranking = ranking.sort_values(
                by=columnas_orden,
                ascending=ascending,
            )


        ranking = ranking.reset_index(
            drop=True
        )


        ranking.insert(
            0,
            "ranking",
            range(
                1,
                len(ranking) + 1,
            ),
        )


        columnas = [
            columna
            for columna in [
                "ranking",
                "material_id",
                "formula_pretty",
                "working_ion",
                "framework_formula",
                "pred_average_voltage",
                "pred_max_delta_volume",
                "energy_above_hull",
                "is_stable",
            ]
            if columna in ranking.columns
        ]


        st.dataframe(
            ranking[columnas],
            use_container_width=True,
            hide_index=True,
        )


        # Descargar CSV
        csv = ranking.to_csv(
            index=False
        ).encode(
            "utf-8"
        )


        st.download_button(
            label="⬇️ Descargar candidatos filtrados",
            data=csv,
            file_name="candidatos_dashboard.csv",
            mime="text/csv",
        )


        st.divider()
        
        st.subheader(
            "🔬 Inspección individual"
        )


        if (
            "material_id" in ranking.columns
            and "formula_pretty" in ranking.columns
        ):

            opciones = ranking.apply(
                lambda fila:
                    f"{fila['material_id']} — "
                    f"{fila['formula_pretty']}",
                axis=1,
            )


        elif "material_id" in ranking.columns:

            opciones = ranking[
                "material_id"
            ].astype(str)


        else:

            opciones = ranking.index.astype(
                str
            )


        seleccionado = st.selectbox(
            "Selecciona un material",
            opciones,
        )


        posicion = list(
            opciones
        ).index(
            seleccionado
        )


        material = ranking.iloc[
            posicion
        ]


        c1, c2, c3, c4 = st.columns(
            4
        )


        if "formula_pretty" in material.index:

            c1.metric(
                "Fórmula",
                str(
                    material[
                        "formula_pretty"
                    ]
                ),
            )


        if "working_ion" in material.index:

            c2.metric(
                "Working ion",
                str(
                    material[
                        "working_ion"
                    ]
                ),
            )


        if "pred_average_voltage" in material.index:

            c3.metric(
                "Voltaje predicho",
                f"{material['pred_average_voltage']:.3f} V",
            )


        if "pred_max_delta_volume" in material.index:

            c4.metric(
                "Cambio de volumen",
                f"{material['pred_max_delta_volume']:.4f}",
            )


        c5, c6, c7 = st.columns(
            3
        )


        if "material_id" in material.index:

            c5.metric(
                "Materials Project ID",
                str(
                    material[
                        "material_id"
                    ]
                ),
            )


        if "energy_above_hull" in material.index:

            c6.metric(
                "Energy above hull",
                f"{material['energy_above_hull']:.4f} eV/atom",
            )


        if "is_stable" in material.index:

            estable = material[
                "is_stable"
            ]

            if isinstance(
                estable,
                str,
            ):

                estable_texto = estable

            else:

                estable_texto = (
                    "Sí"
                    if bool(estable)
                    else "No"
                )


            c7.metric(
                "Estable",
                estable_texto,
            )


        if "framework_formula" in material.index:

            st.write(
                "**Framework:**",
                material[
                    "framework_formula"
                ],
            )


        if "material_id" in material.index:

            url_material = (
                "https://materialsproject.org/materials/"
                f"{material['material_id']}"
            )


            st.link_button(
                "🔗 Ver en Materials Project",
                url_material,
            )


        with st.expander(
            "Ver todos los datos del material"
        ):

            st.dataframe(
                material.to_frame(
                    name="Valor"
                ),
                use_container_width=True,
            )


# 3 - EXPLORACIÓN

with tab3:

    st.header(
        "📊 Exploración de las predicciones"
    )

    # VOLTAJE

    if "pred_average_voltage" in screening.columns:

        st.subheader(
            "Distribución del voltaje predicho"
        )


        histogram_kwargs = {
            "data_frame": screening,
            "x": "pred_average_voltage",
            "nbins": 40,
            "labels": {
                "pred_average_voltage":
                    "Voltaje predicho (V)",
                "working_ion":
                    "Working ion",
            },
            "title":
                "Distribución del voltaje predicho",
        }


        if "working_ion" in screening.columns:

            histogram_kwargs[
                "color"
            ] = "working_ion"


        fig_voltage = px.histogram(
            **histogram_kwargs
        )


        st.plotly_chart(
            fig_voltage,
            use_container_width=True,
        )


    # WORKING ION
  
    if "working_ion" in screening.columns:

        st.subheader(
            "Distribución por working ion"
        )


        ion_counts = (
            screening[
                "working_ion"
            ]
            .value_counts()
            .rename_axis(
                "working_ion"
            )
            .reset_index(
                name="n_materiales"
            )
        )


        fig_iones = px.bar(
            ion_counts,
            x="working_ion",
            y="n_materiales",
            labels={
                "working_ion":
                    "Working ion",
                "n_materiales":
                    "Número de materiales",
            },
            title="Distribución por working ion",
        )


        fig_iones.update_traces(
            texttemplate="%{y}",
            textposition="outside",
        )


        fig_iones.update_layout(
            showlegend=False
        )


        st.plotly_chart(
            fig_iones,
            use_container_width=True,
        )


    # CAMBIO DE VOLUMEN

    if "pred_max_delta_volume" in screening.columns:

        st.subheader(
            "Distribución del cambio de volumen predicho"
        )


        fig_volume = px.histogram(
            screening,
            x="pred_max_delta_volume",
            nbins=40,
            labels={
                "pred_max_delta_volume":
                    "Cambio de volumen predicho",
            },
            title=(
                "Distribución del cambio de volumen predicho"
            ),
        )


        st.plotly_chart(
            fig_volume,
            use_container_width=True,
        )


    # DATASET COMPLETO

    with st.expander(
        "Ver dataset completo de screening"
    ):

        st.dataframe(
            screening,
            use_container_width=True,
            hide_index=True,
        )



# 4 - MODELOS

with tab4:

    st.header(
        "🧠 Comparación de modelos"
    )


    st.markdown(
        """
        Selecciona la variable objetivo y la métrica
        para comparar el rendimiento de los modelos.
        """
    )


    # RESULTADOS VOLTAJE
    
    resultados_voltage = pd.DataFrame(
        {
            "Modelo": [
                "Dummy",
                "Ridge",
                "Random Forest",
                "XGBoost",
            ],
            "MAE": [
                1.387,
                0.753,
                0.596,
                0.562,
            ],
            "RMSE": [
                1.880,
                1.323,
                1.232,
                1.200,
            ],
            "R2": [
                -0.0004,
                0.504,
                0.570,
                0.592,
            ],
        }
    )



    # RESULTADOS VOLUMEN
   
    resultados_volume = pd.DataFrame(
        {
            "Modelo": [
                "XGB composición + log1p",
                "GINE_distance",
            ],
            "MAE": [
                0.404,
                0.376,
            ],
            "RMSE": [
                3.046,
                2.907,
            ],
            "R2": [
                -0.024,
                0.068,
            ],
        }
    )


    # SELECTORES

    variable_objetivo = st.selectbox(
        "Variable objetivo",
        [
            "average_voltage",
            "max_delta_volume",
        ],
    )


    metrica = st.radio(
        "Métrica",
        [
            "MAE",
            "RMSE",
            "R2",
        ],
        horizontal=True,
    )


    st.divider()


  
    # ELEGIR RESULTADOS


    if variable_objetivo == "average_voltage":

        resultados = resultados_voltage.copy()

        titulo_variable = (
            "Average Voltage"
        )

        modelo_final = (
            "XGBoost"
        )


    else:

        resultados = resultados_volume.copy()

        titulo_variable = (
            "Max Delta Volume"
        )

        modelo_final = (
            "GINE_distance"
        )



    # MÉTRICAS EN TARJETAS
  

    st.subheader(
        f"Resultados para {titulo_variable}"
    )


    columnas_metricas = st.columns(
        len(resultados)
    )


    for i, (_, fila) in enumerate(
        resultados.iterrows()
    ):

        columnas_metricas[
            i
        ].metric(
            label=fila["Modelo"],
            value=f"{fila[metrica]:.3f}",
        )


    st.divider()



    # ORDEN


    if metrica in [
        "MAE",
        "RMSE",
    ]:

        resultados_ordenados = (
            resultados
            .sort_values(
                metrica,
                ascending=True,
            )
        )


    else:

        resultados_ordenados = (
            resultados
            .sort_values(
                metrica,
                ascending=False,
            )
        )


    # COLOR DE LA MÉTRICA


    color_metrica = COLORES_METRICAS[
        metrica
    ]



    # GRÁFICO


    fig_metricas = px.bar(
        resultados_ordenados,
        x="Modelo",
        y=metrica,
        text=metrica,
        title=(
            f"{'R²' if metrica == 'R2' else metrica}"
            f" — {titulo_variable}"
        ),
    )


    fig_metricas.update_traces(
        marker_color=color_metrica,
        texttemplate="%{text:.3f}",
        textposition="outside",
    )


    fig_metricas.update_layout(
        height=500,
        yaxis_title=(
            "R²"
            if metrica == "R2"
            else metrica
        ),
        xaxis_title="Modelo",
        showlegend=False,
    )


    st.plotly_chart(
        fig_metricas,
        use_container_width=True,
    )



    # EXPLICACIÓN


    if metrica == "MAE":

        st.info(
            "MAE (Mean Absolute Error) representa el error "
            "absoluto medio de las predicciones. "
            "Cuanto menor sea, mejor."
        )


    elif metrica == "RMSE":

        st.info(
            "RMSE (Root Mean Squared Error) penaliza "
            "especialmente los errores grandes. "
            "Cuanto menor sea, mejor."
        )


    else:

        st.info(
            "R² indica qué proporción de la variabilidad "
            "del target es explicada por el modelo. "
            "Cuanto más próximo a 1, mejor."
        )



    # MEJOR MODELO
  

    if metrica in [
        "MAE",
        "RMSE",
    ]:

        mejor_modelo = resultados.loc[
            resultados[
                metrica
            ].idxmin()
        ]


    else:

        mejor_modelo = resultados.loc[
            resultados[
                metrica
            ].idxmax()
        ]


    nombre_metrica = (
        "R²"
        if metrica == "R2"
        else metrica
    )


    st.success(
        f"🏆 Mejor modelo según {nombre_metrica}: "
        f"{mejor_modelo['Modelo']} "
        f"({nombre_metrica} = "
        f"{mejor_modelo[metrica]:.3f})"
    )


    # TABLA COMPLETA
  

    with st.expander(
        "Ver todas las métricas"
    ):

        st.dataframe(
            resultados,
            use_container_width=True,
            hide_index=True,
            column_config={
                "MAE":
                    st.column_config.NumberColumn(
                        "MAE",
                        format="%.3f",
                    ),
                "RMSE":
                    st.column_config.NumberColumn(
                        "RMSE",
                        format="%.3f",
                    ),
                "R2":
                    st.column_config.NumberColumn(
                        "R²",
                        format="%.3f",
                    ),
            },
        )


    st.caption(
        f"Modelo finalmente utilizado en el pipeline: "
        f"{modelo_final}"
    )



st.divider()

st.caption(
    "TFM · Análisis y predicción de materiales para baterías "
    "mediante aprendizaje automático"
)
