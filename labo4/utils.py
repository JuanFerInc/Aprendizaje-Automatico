estructura = {
    'DEM_RACEETH': {
        'descripcion': "Race",
        'dominio': {
            -9: "Refused",
            -8: "Don't know",
            -4: "Error",
            -3: "Restricted access",
            -1: "Inapplicable",
            # ok
            1: "White non-Hispanic",
            2: "Black non-Hispanic",
            3: "Hispanic",
            4: "Other non-Hispanic",
        }
    },
    'RELIG_IMPORT': {
        'descripcion': "Religion",
        'dominio': { 
            -9: "Refused",
            -8: "Don't know",
            -6: "Not asked, unit nonresponse",
            -4: "Error",
            -3: "Restricted access",
            -1: "Inapplicable",
            # ok
            1: "Important",
            2: "Not important",
        }
    },
    'INCGROUP_PREPOST': {
        'descripcion': "Income",
        'dominio': {
            -9: "Refused",
            -8: "Don't know",
            -7: "Deleted due to partial (Post-election) interview",
            -6: "Not asked, unit nonresponse (no Post-election interview)",
            -4: "Error",
            -3: "Restricted access",
            -1: "Inapplicable",
            # ok
            1: "Under $5,000",
            2: "$5,000-$9,999",
            3: "$10,000-$12,499",
            4: "$12,500-$14,999",
            5: "$15,000-$17,499",
            6: "$17,500-$19,999",
            7: "$20,000-$22,499",
            8: "$22,500-$24,999",
            9: "$25,000-$27,499",
            10: "$27,500-$29,999",
            11: "$30,000-$34,999",
            12: "$35,000-$39,999",
            13: "$40,000-$44,999",
            14: "$45,000-$49,999",
            15: "$50,000-$54,999",
            16: "$55,000-$59,999",
            17: "$60,000-$64,999",
            18: "$65,000-$69,999",
            19: "$70,000-$74,999",
            20: "$75,000-$79,999",
            21: "$80,000-$89,999",
            22: "$90,000-$99,999",
            23: "$100,000-$109,999",
            24: "$110,000-$124,999",
            25: "$125,000-$149,999",
            26: "$150,000-$174,999",
            27: "$175,000-$249,999",
            28: "$250,000+",
        }
    },
    'DEM_EDUGROUP': {
        'descripcion': "Education",
        'dominio': {
            -9: "Refused",
            -8: "Don't know",
            -4: "Error",
            -3: "Restricted access",
            -2: "Missing, other not codeable to 1-5",
            -1: "Inapplicable",
            # ok
            1: "Less than high school credential",
            2: "High school credential",
            3: "Some post-high-school, no bachelor's degree",
            4: "Bachelor's degree",
            5: "Graduate degree",
        }
    },
    'DEM_AGEGRP_IWDATE': {
        'descripcion': "Age",
        'dominio': {
            -9: "Refused",
            -8: "Don't know",
            -4: "Error",
            -3: "Restricted access",
            -2: "Missing, birthdate fields left blank",
            -1: "Inapplicable",
            # ok
            1: "17-20",
            2: "21-24",
            3: "25-29",
            4: "30-34",
            5: "35-39",
            6: "40-44",
            7: "45-49",
            8: "50-54",
            9: "55-59",
            10: "60-64",
            11: "65-69",
            12: "70-74",
            13: "75+",
        }
    },
    'LIBCPRE_SELF': {
        'descripcion': "Self placement",
        'dominio': {
            -9: "Refused",
            -8: "Don't know",
            -6: "Not asked, unit nonresponse",
            -4: "Error",
            -3: "Restricted access",
            -2: "Haven't thought much about this",
            -1: "Inapplicable",
            # ok
            1: "Extremely liberal",
            2: "Liberal",
            3: "Slightly liberal",
            4: "Moderate; middle of the road",
            5: "Slightly conservative",
            6: "Conservative",
            7: "Extremely conservative",
        }
    }
}

columnas = estructura.keys()

""" retorna true si col es categorica (~tiene dominio discreto) """ 
def esCategorica(col):
    # como todas las columnas son categoricas
    # return col in columnas
    return True

"""
retorna el dominio de col
PRE: col es categorica
"""
def dominio(col):
    return list(estructura[col]['dominio'].keys())

"""
retorna solo el dominio positivo de col
PRE: col es categorica
"""
def dominio_positivo(col):
    return [val for val in dominio('LIBCPRE_SELF') if val > 0]