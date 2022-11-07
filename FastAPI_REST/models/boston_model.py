from pydantic import BaseModel, Field


class BostonModel(BaseModel):
    INDICE_CRIMEN: float = Field()
    PCT_ZONA_RESIDENCIAL: float = Field()
    PCT_ZONA_INDUSTRIAL: float = Field()
    RIO_CHARLES: int = Field()
    OXIDO_NITROSO_PPM: float = Field()
    N_HABITACIONES_MEDIO: float = Field()
    PCT_CASAS_40S: float = Field()
    DIS: float = Field()
    DIS_AUTOPISTAS: int = Field()
    CARGA_FISCAL: int = Field()
    RATIO_PROFESORES: float = Field()
    PCT_NEGRA: float = Field()
    PCT_CLASE_BAJA: float = Field()

    class Config:
        schema_extra = {
            'example': {
                'INDICE_CRIMEN': 0.00632,
                'PCT_ZONA_RESIDENCIAL': 18.0,
                'PCT_ZONA_INDUSTRIAL': 2.31,
                'RIO_CHARLES': 0,
                'OXIDO_NITROSO_PPM': 0.538,
                'N_HABITACIONES_MEDIO': 6.575,
                'PCT_CASAS_40S': 65.2,
                'DIS': 4.0900,
                'DIS_AUTOPISTAS': 1,
                'CARGA_FISCAL': 296,
                'RATIO_PROFESORES': 15.3,
                'PCT_NEGRA': 396.90,
                'PCT_CLASE_BAJA': 4.98
            }
        }