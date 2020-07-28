

import json
from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError
from jsonschema.exceptions import SchemaError



#######################################################################################
#
#   exec
# 

def exec(p, schema):

    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as e:
        print(str(e))
        return

    v = Draft7Validator(schema)

    try:
        v.validate(p)
    except ValidationError as e:
        print (str(e))

# PERIODE - BESKR/ÅRSAK

beskr_arsak = {
    "type":"object",
    "properties": {
        "beskrivelse": {"type":["string", "null"]},
        "arsak": {"type":"array"}},
    "required" : ["beskrivelse", "arsak"]
}

# PERIODE aktivitet ikke mulig

aktivitet_ikke_mulig = {

    "type" : "object",

    "oneOf": [
        {"properties": { "medisinskArsak":   beskr_arsak}},
        {"properties": { "medisinskArsak":  { "type": "null"}}},
    ],  
    
    "oneOf": [
        {"properties": { "arbeidsrelatertArsak":   beskr_arsak}},
        {"properties": { "arbeidsrelatertArsak":  { "type": "null"}}},
    ],
   
    "required": ["medisinskArsak", "arbeidsrelatertArsak"]
}

# PERIODE  -  gradert

gradert = {
    "type" : "object",
    "properties":
    {
        "reisetilskudd": {"type":"boolean"},
        "grad": {"type":"integer"}
    },
    "required" : ["reisetilskudd", "grad"]
}

# PERIODE - complete

periode_schema = {

    "type": "object",
    "oneOf": [
        {"properties": { "aktivitetIkkeMulig":                      aktivitet_ikke_mulig}},
        {"properties": { "avventendeInnspillTilArbeidsgiver":       { "type": ["string"] }}},
        {"properties": { "behandlingsdager":                        { "type": ["integer"] }}},
        {"properties": { "gradert":                                 gradert}},
        {"properties": { "reisetilskudd":                           { "type": "boolean", "const": True}}}
    ],
    "properties": {
        "fom" : {"type": "string"},
        "tom" : {"type": "string"}
    },
    "required": ["fom", "tom"]
    
}

diagnose = {
    "type" : "object",
    "properties" : {
        "system" : {"type": "string"},
        "kode"   : {"type": "string"},
        "tekst"  : {"type": "string"}
        },
    "required" : ["system", "kode", "tekst"]
    }


medisinskVurdering = {
    "type" : "object",
    "properties" : {"hovedDiagnose" : diagnose,
                    "biDiagnoser": {"type": "array", "items" : diagnose, "minItems": 0},

                    "svangerskap" : {"type": "boolean"},
                    "yrkesskade"  : {"type": "boolean"},
                    "yrkesskadeDato"  : {"type": ["null", "string"]},
                    "annenFraversArsak" : {"type" : ["null", "string"]}

                    },
    "required" : ["hovedDiagnose", "biDiagnoser", "svangerskap", "yrkesskade", "yrkesskadeDato"]
    
    }

s['medisinskVurdering']

exec(s['medisinskVurdering'], medisinskVurdering)


# SYKMELDING COMPLETE


sykmelding_schema = {
    "type": "object",
    "properties" : {
                    "perioder": { "type": "array", "items":  periode_schema, "minItems": 1},
                    "medisinskVurdering" : medisinskVurdering                  
                    },
    "required" : ["perioder"]
    }


...

exec(s, sykmelding_schema)


exec(p_p, periode_schema)

exec(l[2], aktivitet_ikke_mulig)
