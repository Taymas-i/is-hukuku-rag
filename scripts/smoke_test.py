import logging
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
from app.config import settings
from app.rag.graph import build_initial_state, get_rag_graph

QS = [
    ("5 yıldır çalışıyorum, kaç gün yıllık iznim var?", "14 gün"),
    ("10 yıllık çalışanın yıllık izni kaç gündür?", "20 gün"),
    ("İşverenim beni haber vermeden işten çıkardı, ne yapabilirim?", "m.17 ihbar"),
    ("Haftada en fazla kaç saat çalıştırılabilirim?", "45 saat"),
    ("Doğum izni kaç hafta?", "16 hafta (m.74)"),
    ("Fazla mesai ücreti yüzde kaç zamlı ödenir?", "%50 (m.41)"),
    ("Deneme süresi en fazla ne kadar olabilir?", "2 ay, TİS ile 4 ay (m.15)"),
    ("Hırsızlığın cezası nedir?", "RED"),
]
print("MODEL:", settings.LLM_MODEL)
g = get_rag_graph()
g.invoke(build_initial_state("ısınma sorusu yıllık izin"))
for q, exp in QS[: int(sys.argv[1]) if len(sys.argv) > 1 else None]:
    t = time.perf_counter()
    out = g.invoke(build_initial_state(q))
    print(f"\n[{time.perf_counter() - t:.1f}s] {q}  (beklenen: {exp})")
    print(f"  {out['final_status']} gen={out['generation_count']} rw={out['rewrite_retries']} top={[round(s, 2) for s in out['retrieved_scores'][:3]]}")
    print("  ", out["answer"].replace("\n", " ")[:400])
