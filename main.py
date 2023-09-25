if __name__ == "__main__":
    import uvicorn
    import os

    workers = int(os.environ.get("FACESWAP_WORKERS", 1))
    print(f"Starting server with {workers} workers")
    uvicorn.run("app:app", host="0.0.0.0", port=8112, reload=False, workers=workers)
