import re
import json
import spacy
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path


class HybridMasker:
    """Hybrid PII Masker using Regex + spaCy NER.

    Inputs:
    patterns_dct: Regex patterns by entity type
    ner_entities_lst: spaCy entity labels to mask
    Outputs:
    result_dct: Masked text + entities + stats
    """


    def __init__(
            self,
            patterns_dct: Dict[str, List[str]],
            ner_entities_lst: List[str],
            spacy_model_str: str,
            masking_mode_str: str,
        ):

        self.patterns_dct = {k: [re.compile(p) for p in v] for k, v in patterns_dct.items()}
        self.ner_entities_lst = ner_entities_lst
        self.masking_mode_str = masking_mode_str.lower()
        self.nlp = None
        
        if self.masking_mode_str not in {"regex", "ner", "hybrid"}:
            raise ValueError(f"Invalid masking_mode: {self.masking_mode_str}")
        
        if self.masking_mode_str in {"ner", "hybrid"}:
            self.nlp = spacy.load(spacy_model_str)


    def mask(self, text: str) -> Dict[str, Any]:
        entity_lst: List[Dict[str, Any]] = []

        # Regex masking
        if self.masking_mode_str in {"regex", "hybrid"}:
            for ent_type, pat_lst in self.patterns_dct.items():
                for pat in pat_lst:
                    for m in pat.finditer(text):
                        entity_lst.append({
                        "type": ent_type,
                        "start": m.start(),
                        "end": m.end(),
                        "value": text[m.start():m.end()],
                    })

        # spaCy NER masking
        if self.masking_mode_str in {"ner", "hybrid"} and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in self.ner_entities_lst:
                    entity_lst.append({
                        "type": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "value": ent.text,
                    })


        # Resolve overlaps
        final_entities_lst: List[Dict[str, Any]] = []
        for cand in sorted(entity_lst, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
            if not any(cand["start"] < e["end"] and cand["end"] > e["start"] for e in final_entities_lst):
                final_entities_lst.append(cand)


        # Apply masking
        masked_text: str = text
        for ent in sorted(final_entities_lst, key=lambda x: x["start"], reverse=True):
            masked_text = masked_text[:ent["start"]] + f"[{ent['type']}]" + masked_text[ent["end"]:]


        stats_dct: Dict[str, int] = {}
        for e in final_entities_lst:
            stats_dct[e["type"]] = stats_dct.get(e["type"], 0) + 1


        return {
            "masked_text": masked_text,
            "entities": final_entities_lst,
            "stats": stats_dct,
        }


def load_config(config_path: Path) -> Dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def main() -> None:
    BaseDir: Path = Path(__file__).resolve().parent


    InpRelPath: Path = Path("input")
    InpAbsPath: Path = BaseDir / InpRelPath


    ResRelPath: Path = Path("results")
    ResAbsPath: Path = BaseDir / ResRelPath
    ResAbsPath.mkdir(exist_ok=True)


    CfgAbsPath: Path = BaseDir / "config" / "masking_config.json"


    cfg_dct: Dict[str, Any] = load_config(CfgAbsPath)


    masker_ins = HybridMasker(
        patterns_dct=cfg_dct["regex_patterns"],
        ner_entities_lst=cfg_dct["ner_entities"],
        spacy_model_str=cfg_dct["spacy_model"],
        masking_mode_str=cfg_dct["masking_mode"]
    )


    sample_text: str = """
    Criteria for Dependent Visa Sponsorship at Dentsu
    Dependents are defined as the employee’s spouse and/or unmarried children under 21 years old.
    Dentsu may cover the costs for dependents’ visa status or work authorization, but this is decided on a case-by-case basis.
    If Dentsu chooses not to cover dependent costs, the employee will be notified as soon as possible so they can prepare the necessary documents.
    Dependents can still use Dentsu’s immigration attorney at the same annual fee rate negotiated for the company, even if Dentsu does not cover their costs.
    For visa stamping at a U.S. Consulate, Dentsu will cover the costs for dependents only in certain cases (e.g., Blanket L, E-3, TN, H-1B1), but not for travel expenses.
    If a visa stamp is needed after USCIS approval, Dentsu will not cover consulate visa costs for dependents, but the discounted attorney rate still applies1.
    """


    result_dct: Dict[str, Any] = masker_ins.mask(sample_text)


    ts_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path: Path = ResAbsPath / f"masked_result_{ts_str}.json"
    out_path.write_text(json.dumps(result_dct, indent=2), encoding="utf-8")


    print("Masked Output:", result_dct["masked_text"])




if __name__ == "__main__":
    main()