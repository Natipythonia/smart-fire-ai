import { useEffect, useState } from 'react'
import './App.css'

type Deteccion = {
  clase: string
  puntuacion: number
  caja: number[]
}

type Resultado = {
  modelo: string
  ancho: number
  alto: number
  umbral: number
  tiempo_inferencia_segundos: number
  total_detecciones: number
  detecciones: Deteccion[]
  imagen_analizada: string
  aviso: string
}

function App() {
  const [archivo, setArchivo] = useState<File | null>(null)
  const [vistaPrevia, setVistaPrevia] = useState<string | null>(null)
  const [resultado, setResultado] = useState<Resultado | null>(null)
  const [analizando, setAnalizando] = useState(false)
  const [error, setError] = useState('')
  const [apiConectada, setApiConectada] = useState(false)

  useEffect(() => {
    fetch('/api/health')
      .then((respuesta) => {
        if (!respuesta.ok) throw new Error('API no disponible')
        setApiConectada(true)
      })
      .catch(() => setApiConectada(false))
  }, [])

  useEffect(() => {
    return () => {
      if (vistaPrevia) URL.revokeObjectURL(vistaPrevia)
    }
  }, [vistaPrevia])

  function seleccionarArchivo(evento: React.ChangeEvent<HTMLInputElement>) {
    const seleccionado = evento.target.files?.[0] ?? null
    const nuevaVistaPrevia = seleccionado
      ? URL.createObjectURL(seleccionado)
      : null

    setVistaPrevia(nuevaVistaPrevia)
    setArchivo(seleccionado)
    setResultado(null)
    setError('')
  }

  async function analizarImagen() {
    if (!archivo || analizando) return

    setAnalizando(true)
    setError('')
    setResultado(null)

    try {
      const datos = new FormData()
      datos.append('archivo', archivo)

      const respuesta = await fetch('/api/detect/smoke', {
        method: 'POST',
        body: datos,
      })

      if (!respuesta.ok) {
        let mensaje = 'No se pudo analizar la fotografía.'

        try {
          const cuerpo = await respuesta.json()
          if (typeof cuerpo.detail === 'string') mensaje = cuerpo.detail
        } catch {
          // Se conserva el mensaje general.
        }

        throw new Error(mensaje)
      }

      const informacion: Resultado = await respuesta.json()
      setResultado(informacion)
      setApiConectada(true)
    } catch (fallo) {
      setError(
        fallo instanceof Error
          ? fallo.message
          : 'Ha ocurrido un error inesperado.',
      )
    } finally {
      setAnalizando(false)
    }
  }

  return (
    <div className="app">
      <header className="cabecera">
        <div className="marca">
          <div className="marca-icono">🔥</div>
          <div>
            <h1>SMART FIRE AI</h1>
            <p>ATLANTIA TECH · Inteligencia artificial aplicada a incendios forestales</p>
          </div>
        </div>

        <div className="estado-superior">
          <span className="etiqueta-experimental">MVP 1.0 · EXPERIMENTAL</span>
          <span className={apiConectada ? 'api-ok' : 'api-error'}>
            ● {apiConectada ? 'API conectada' : 'API no disponible'}
          </span>
        </div>
      </header>

      <main className="contenido">
        <section className="presentacion">
          <div className="seccion-indicador">MÓDULO 01 / VISIÓN ARTIFICIAL</div>
          <h2>Detección inteligente de humo</h2>
          <p>
            Analiza fotografías forestales mediante inteligencia artificial
            e identifica posibles regiones con humo para su revisión.
          </p>
        </section>

        <section className="paneles">
          <div className="panel">
            <div className="panel-encabezado">
              <h3>Imagen de origen</h3>
              <span>ENTRADA</span>
            </div>

            <label className="zona-carga">
              <span className="icono-carga">↑</span>
              <strong>Seleccionar fotografía</strong>
              <small>JPG, PNG o WebP · Máximo 10 MB</small>
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={seleccionarArchivo}
                disabled={analizando}
              />
            </label>

            {vistaPrevia && (
              <div className="contenedor-imagen">
                <img src={vistaPrevia} alt="Fotografía seleccionada" />
              </div>
            )}

            {archivo && <p className="nombre-archivo">{archivo.name}</p>}

            <button
              className="boton-analizar"
              onClick={analizarImagen}
              disabled={!archivo || analizando || !apiConectada}
            >
              {analizando ? 'Analizando fotografía...' : 'Analizar con IA'}
            </button>
          </div>

          <div className="panel">
            <div className="panel-encabezado">
              <h3>Resultado de detección</h3>
              <span>SALIDA IA</span>
            </div>

            {!resultado && !error && (
              <div className="resultado-vacio">
                <span>◎</span>
                <strong>Esperando análisis</strong>
                <p>Selecciona una fotografía y pulsa «Analizar con IA».</p>
              </div>
            )}

            {error && <div className="mensaje-error">{error}</div>}

            {resultado && (
              <>
                <div className="contenedor-imagen">
                  <img
                    src={resultado.imagen_analizada}
                    alt="Fotografía analizada con posibles detecciones"
                  />
                </div>

                <div className="metricas">
                  <div>
                    <small>Detecciones</small>
                    <strong>{resultado.total_detecciones}</strong>
                  </div>
                  <div>
                    <small>Tiempo IA</small>
                    <strong>{resultado.tiempo_inferencia_segundos} s</strong>
                  </div>
                  <div>
                    <small>Umbral</small>
                    <strong>{Math.round(resultado.umbral * 100)} %</strong>
                  </div>
                </div>

                {resultado.total_detecciones === 0 ? (
                  <div className="mensaje-neutro">
                    No se han identificado regiones con humo por encima del umbral.
                    Esto no garantiza la ausencia de incendio.
                  </div>
                ) : (
                  <div className="detecciones">
                    <h4>Regiones sospechosas</h4>
                    {resultado.detecciones.map((deteccion, indice) => (
                      <div className="deteccion" key={indice}>
                        <span>Posible humo #{indice + 1}</span>
                        <strong>{(deteccion.puntuacion * 100).toFixed(1)} %</strong>
                      </div>
                    ))}
                  </div>
                )}

                <a
                  className="enlace-descarga"
                  href={resultado.imagen_analizada}
                  download="smart_fire_ai_resultado.jpg"
                >
                  Descargar imagen analizada ↓
                </a>
              </>
            )}
          </div>
        </section>

        <section className="aviso">
          <strong>⚠ Herramienta experimental de investigación</strong>
          <p>
            Smart Fire AI MVP 1.0 no constituye un sistema certificado de
            detección de incendios. Los resultados necesitan verificación
            humana y no sustituyen a los servicios oficiales de emergencias.
          </p>
        </section>

        <footer>
          ATLANTIA TECH · SMART FIRE AI · MVP 1.0
        </footer>
      </main>
    </div>
  )
}

export default App
