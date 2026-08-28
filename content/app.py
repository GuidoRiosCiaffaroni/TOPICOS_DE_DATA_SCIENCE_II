from fastapi import FastAPI

app = FastAPI(
    title="API de Machine Learning",
    description="API REST ejecutándose dentro de Google Colab",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "API FastAPI funcionando correctamente en Google Colab"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment": "Google Colab",
        "service": "FastAPI",
        "version": "1.0.0"
    }
