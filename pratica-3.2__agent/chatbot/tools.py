def estimate_solar_generation(
    installed_power_kwp: float,
    irradiation_kwh_m2_day: float,
    performance_ratio: float = 0.80,
    days: int = 30,
    emission_factor_kg_kwh: float | None = None
) -> dict:
    """
    Estima a geração de energia solar e a redução de emissões de CO2.

    Args:
        installed_power_kwp: Potência instalada em kWp.
        irradiation_kwh_m2_day: Irradiação solar média diária.
        performance_ratio: Fator de desempenho do sistema (0 a 1).
        days: Período de cálculo em dias.
        emission_factor_kg_kwh: Fator de emissão da rede elétrica.

    Returns:
        Estimativa de geração energética e impacto ambiental.
    """

    if installed_power_kwp <= 0:
        raise ValueError("A potência instalada deve ser positiva.")

    if irradiation_kwh_m2_day < 0:
        raise ValueError("A irradiação não pode ser negativa.")

    if not 0 < performance_ratio <= 1:
        raise ValueError("O fator de desempenho deve estar entre 0 e 1.")

    if days <= 0:
        raise ValueError("O número de dias deve ser positivo.")

    generation = (
        installed_power_kwp
        * irradiation_kwh_m2_day
        * performance_ratio
        * days
    )

    result = {
        "monthly_generation_kwh": round(generation, 2),
        "installed_power_kwp": installed_power_kwp,
        "performance_ratio": performance_ratio,
        "period_days": days
    }

    if emission_factor_kg_kwh is not None:
        if emission_factor_kg_kwh < 0:
            raise ValueError("O fator de emissão não pode ser negativo.")

        result["avoided_co2_kg"] = round(
            generation * emission_factor_kg_kwh, 2
        )

    return result


# Sugestão de mais ferramentas implementadas: 
#   NASA POWER API: obtenção de dados reais de irradiação solar