# Laboratorio: transformación de intensidades a partir del histograma

Esta carpeta es un **proyecto independiente** de la aplicación que aproxima la CDF. Aquí se aproxima directamente el perfil del histograma.

## Qué hace la aplicación

- Carga una imagen y la convierte a escala de grises de 8 bits.
- Calcula y muestra el histograma original y la CDF empírica.
- Aproxima el perfil del histograma normalizado con PCHIP, interpolación lineal o un polinomio trigonométrico de Fourier por mínimos cuadrados.
- Permite modificar la cantidad de puntos de control y, en Fourier, el número de armónicos.
- Permite explorar cuatro parámetros: `A` y `B` modifican la variable; `C` y `D` modifican verticalmente la función.
- Muestra la curva aproximada, la curva transformada, el mapa de intensidades, las distribuciones resultantes, las métricas y las imágenes de comparación con HE y CLAHE.
- Permite descargar la imagen del modelo y las métricas.

## Definición experimental

El histograma se normaliza por su frecuencia máxima para obtener un perfil \(\widehat h(r)\) con escala independiente del tamaño de la imagen. Luego se define:

\[
G(r)=C\,\widehat h(Ar+B)+D,
\]

\[
T(r)=r+\lambda\,[G(r)-\widehat h(r)].
\]

`lambda` (en la interfaz, **Escala del efecto sobre las intensidades**) controla cuánto de la diferencia entre la curva transformada y la aproximación original se traduce en el mapa de intensidades. Con los parámetros neutros `A=1`, `B=0`, `C=1`, `D=0`, se tiene \(G(r)=\widehat h(r)\), por lo que \(T(r)=r\): la imagen debe permanecer idéntica a la original.

**Importante:** las frecuencias del histograma no son intensidades de salida. Por eso se explicita una regla experimental para convertir la deformación de la curva en desplazamientos de intensidad. Esta regla es parte del modelo que se explora y no equivale a la ecualización clásica. Algunas combinaciones pueden producir mapas no monótonos o recortes en 0 y 255; se muestran indicadores para identificar esos casos.

## Ejecutar localmente (opcional)

Requiere Python 3.10 o posterior.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Subirlo a GitHub

1. Descomprime el ZIP.
2. En GitHub, crea un repositorio nuevo, por ejemplo `laboratorio-transformacion-histograma`.
3. En el repositorio, entra a **Add file → Upload files**.
4. Sube `streamlit_app.py`, `requirements.txt`, `README.md` y `.gitignore` (no subas solo el ZIP).
5. Pulsa **Commit changes**.

## Publicarlo en Streamlit Community Cloud

1. Abre <https://share.streamlit.io/> e inicia sesión con GitHub.
2. Elige **Create app**.
3. Selecciona el repositorio, la rama `main` y el archivo principal `streamlit_app.py`.
4. Pulsa **Deploy**.

## Estructura

```text
Proyecto_Streamlit_Histograma/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```
