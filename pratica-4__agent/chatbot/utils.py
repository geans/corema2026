from datetime import date, timedelta

import httpx
from agno.tools import tool


@tool
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
@tool
def get_solar_irradiation(
    latitude: float,
    longitude: float,
) -> dict:
    """
    Consulta a NASA POWER API para obter a irradiação solar média diária
    de uma localização considerando o histórico dos últimos 365 dias.

    Utilize esta ferramenta quando o usuário perguntar sobre:
    - Irradiação solar média diária.
    - Potencial de geração de energia solar.
    - Disponibilidade de radiação solar em determinada região.
    - Condições históricas de radiação solar de uma localização.

    A consulta utiliza o parâmetro ALLSKY_SFC_SW_DWN da NASA POWER,
    que representa a irradiação solar global incidente na superfície,
    considerando condições reais de céu.

    O período consultado é automaticamente definido como os últimos
    365 dias, considerando a data atual.

    Args:
        latitude (float): Latitude da localização em graus decimais.
            Valores negativos representam o hemisfério sul.
        longitude (float): Longitude da localização em graus decimais.
            Valores negativos representam o hemisfério oeste.

    Returns:
        dict: Resultado contendo:
            - latitude: Latitude consultada.
            - longitude: Longitude consultada.
            - start_date: Data inicial da consulta.
            - end_date: Data final da consulta.
            - average_daily_irradiation: Irradiação média diária.
            - unit: Unidade de medida.
            - valid_days: Quantidade de dias válidos utilizados.
            - parameter: Parâmetro consultado.
            - source: Fonte dos dados.

    Raises:
        ValueError: Quando não há dados válidos.
        httpx.HTTPError: Quando ocorre falha na consulta à API.
    """

    end = date.today()
    start = end - timedelta(days=365)

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "latitude": latitude,
        "longitude": longitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }

    response = httpx.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    values = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

    valid_values = [
        value
        for value in values.values()
        if value is not None and value != -999
    ]

    if not valid_values:
        raise ValueError(
            "A NASA POWER não retornou dados válidos para o período."
        )

    average = sum(valid_values) / len(valid_values)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "average_daily_irradiation": round(average, 2),
        "unit": "kWh/m²/day",
        "valid_days": len(valid_values),
        "parameter": "ALLSKY_SFC_SW_DWN",
        "source": "NASA POWER",
    }


@tool
def get_coordinates(address: str) -> dict:
    """
    Obtém latitude e longitude a partir de um endereço ou parte dele,
    utilizando a API de geocodificação do OpenStreetMap (Nominatim).

    Args:
        address: Endereço completo ou parcial para pesquisa.
                 Exemplo: "Universidade Federal de Alagoas, Maceió"

    Returns:
        Dicionário contendo latitude, longitude e endereço encontrado.

    Raises:
        ValueError: Se nenhum endereço correspondente for encontrado.
        httpx.HTTPError: Se ocorrer algum erro na requisição.
    """

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": address,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }

    headers = {
        "User-Agent": "SolarEnergyAgent/1.0 (Python)"
    }

    response = httpx.get(
        url,
        params=params,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        raise ValueError(f"Nenhum endereço encontrado: {address}")

    result = results[0]

    return {
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "address": result["display_name"],
    }


# Exemplo de utilização
if __name__ == "__main__":
    result = get_coordinates(
        "Universidade Federal de Alagoas, Maceió"
    )

    print(result)