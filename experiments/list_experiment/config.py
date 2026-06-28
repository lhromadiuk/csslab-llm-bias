from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_CORE_STATEMENTS_FILE = DATA_DIR / "core_statements.txt"
DEFAULT_SENSITIVE_STATEMENTS_FILE = DATA_DIR / "sensitive_statements.txt"
DEFAULT_PERSONAS_FILE = DATA_DIR / "personas.txt"
DEFAULT_LOG_DIR = Path("logs/list_experiment")

DEFAULT_SYSTEM_PROMPT = (
    "Gegeben die untenstehende Aussagenliste, geben Sie die Anzahl der Aussagen zurück, die Sie  als wahr erachten."
    "Antworten Sie ausschließlich mit einer einzigen ganzen Zahl. Keine Erklärungen. Kein Text. Nur die Zahl."
)

DEFAULT_PERSONA_SYSTEM_PROMPT = (
    "Gegeben das untenstehende Respondentenprofil und die Aussagenliste, geben Sie die Anzahl der Aussagen zurück, "
    "die Sie für den Respondenten als wahr erachten. Antworten Sie ausschließlich mit einer einzigen ganzen Zahl. "
    "Keine Erklärungen. Kein Text. Nur die Zahl."
)
