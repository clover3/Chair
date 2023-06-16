
from contradiction.medical_claims.token_tagging.gpt_solver.get_chat_gpt_solver import get_chat_gpt_file_solver
from contradiction.medical_claims.token_tagging.runner_s.run_exact_match import solve_and_save
from utils.open_ai_api import ENGINE_GPT_3_5


def main():
    tag_type = "mismatch"
    engine = ENGINE_GPT_3_5
    run_name = engine
    solver = get_chat_gpt_file_solver(engine, tag_type)
    solve_and_save(run_name, solver, tag_type)


if __name__ == "__main__":
    main()
