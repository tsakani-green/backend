from app.schemas.esg import ESGInput, ESGScores


def normalize_inverse(value: float, best: float, worst: float) -> float:
    """
    Lower is better: emissions, waste, etc.
    Returns 0–100.
    """
    value_clamped = max(min(value, worst), best)
    if worst == best:
        return 50.0
    score = (worst - value_clamped) / (worst - best) * 100
    return round(score, 2)


def normalize_direct(value: float, best: float, worst: float) -> float:
    """
    Higher is better: social and governance scores.
    """
    value_clamped = max(min(value, best), worst)
    if best == worst:
        return 50.0
    score = (value_clamped - worst) / (best - worst) * 100
    return round(score, 2)


def calculate_esg_scores(data: ESGInput) -> ESGScores:
    # Simple assumptions – adjust with real benchmarks later
    e_components = [
        normalize_inverse(data.carbon_emissions_tons, best=0, worst=5000),
        normalize_inverse(data.energy_consumption_mwh, best=0, worst=10000),
        normalize_inverse(data.water_use_m3, best=0, worst=20000),
        normalize_inverse(data.waste_generated_tons, best=0, worst=1000),
    ]
    e_score = round(sum(e_components) / len(e_components), 2)

    s_score = normalize_direct(data.social_score_raw, best=100, worst=0)
    g_score = normalize_direct(data.governance_score_raw, best=100, worst=0)

    overall = round((0.4 * e_score) + (0.3 * s_score) + (0.3 * g_score), 2)

    methodology = {
        "environmental": "Inverse normalization on emissions, energy, water, waste.",
        "social": "Direct 0–100 score.",
        "governance": "Direct 0–100 score.",
        "weights": "E=40%, S=30%, G=30%.",
    }

    return ESGScores(
        company_name=data.company_name,
        period=data.period,
        e_score=e_score,
        s_score=s_score,
        g_score=g_score,
        overall_score=overall,
        methodology=methodology,
    )
