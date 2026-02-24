import os
import sys

def main() -> None:
    port = os.getenv("PORT", "8501")
    os.environ["STREAMLIT_SERVER_PORT"] = port
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", "app.py"]
    stcli.main()

if __name__ == "__main__":
    main()
