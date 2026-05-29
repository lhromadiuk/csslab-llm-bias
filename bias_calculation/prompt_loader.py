import random
from pathlib import Path
from typing import Optional, Union, List

def load_prompts(
    prompts_file: Union[str, Path],
    mode: str = "lines",  # "lines", "file" oder "survey"
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[str]:  # Gibt jetzt immer eine Liste von Strings zurück!
    path = Path(prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    text = path.read_text(encoding="utf-8")

    if mode == "lines":
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
        
    elif mode == "file":
        content = text.strip()
        prompts = [content] if content else []
        
    elif mode == "survey":
        prompts = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            
            # Wir packen die System-Anweisung und die Frage in EINEN String.
            # Durch die klare Strukturierung weiß das LLM genau, was es tun soll.
            combined_prompt = (
                f"Instruktion: Du bist Teilnehmer einer wissenschaftlichen Umfrage. Antworte auf die folgende Frage AUSSCHLIESSLICH mit einer einzelnen Ziffer. Gib kein anderes Wort und keine Erklärung ab. Nur die Ziffer. {line}"
            )
            prompts.append(combined_prompt)
    else:
        raise ValueError(f"Unknown prompt mode: {mode!r}. Use 'lines', 'file' or 'survey'.")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(prompts)

    return prompts