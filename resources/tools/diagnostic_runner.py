# diagnostic_runner.py - Non-destructive tests for SarahMemory 7.7.1
import sys, json
def main():
    report = {}
    try:
        import SarahMemoryMain as M
        report["main_boot"] = getattr(M, "selftest", lambda: "no_selftest")()
    except Exception as e:
        report["main_boot"] = f"ERROR: {e}"
    try:
        import SarahMemoryIntegration as I
        report["gui_boot"] = "launch_gui_exists" if hasattr(I, "launch_gui") else "launch_gui_missing"
    except Exception as e:
        report["gui_boot"] = f"ERROR: {e}"
    try:
        import SarahMemoryGUI as G
        report["status_lights"] = getattr(G, "_selftest_status_lights", lambda: "missing")()
        report["avatar"] = getattr(G, "selftest_avatar", lambda: "missing")()
    except Exception as e:
        report["gui"] = f"ERROR: {e}"
    try:
        import SarahMemoryLLM as L
        report["llm_self"] = getattr(L, "selftest", lambda: "missing")()
    except Exception as e:
        report["llm_self"] = f"ERROR: {e}"
    try:
        import SarahMemorySystemIndexer as X
        import SarahMemorySystemLearn as S
        report["indexer"] = getattr(X, "cli_index", lambda **k: "missing")(sample_only=True)
        report["learn"] = getattr(S, "cli_learn", lambda **k: "missing")(sample_only=True)
    except Exception as e:
        report["index_learn"] = f"ERROR: {e}"
    try:
        import SarahMemoryVoice as V
        report["wake_words"] = getattr(V, "selftest_wakewords", lambda: "missing")()
    except Exception as e:
        report["wake_words"] = f"ERROR: {e}"
    try:
        import SarahMemoryDiagnostics as D
        report["cuda"] = getattr(D, "selftest_cuda", lambda: "missing")()
    except Exception as e:
        report["cuda"] = f"ERROR: {e}"
    try:
        import SarahMemoryFacialRecognition as F
        report["vision_models"] = getattr(F, "selftest_models", lambda: "missing")()
    except Exception as e:
        report["vision_models"] = f"ERROR: {e}"
    try:
        import SarahMemoryUpdater as U
        report["updater"] = getattr(U, "selftest", lambda: "missing")()
    except Exception as e:
        report["updater"] = f"ERROR: {e}"
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
