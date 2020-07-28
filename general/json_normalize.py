

import json
import pandas as pd


l = []

a = { 'sykmelding': {'id': 'a', 'msgId': '', 'perioder': [{'fom': '2017-01-06', 'tom': '2017-03-05', 'gradert': 11}, {'fom': '2017-02-06', 'tom': '2017-03-05', 'gradert': None}], 'meldingTilNAV': None, 'avsenderSystem': {'navn': 'System X', 'versjon': '3.9.1.3 (5015)'}, 'pasientAktoerId': '1000035002112', 'kontaktMedPasient': {'kontaktDato': None, 'begrunnelseIkkeKontakt': None}, 'behandletTidspunkt': '2017-02-05T23:00:00', 'medisinskVurdering': {'yrkesskade': False, 'biDiagnoser': [{'kode': 'R05', 'tekst': 'HOSTE', 'system': '2.16.578.1.12.4.1.1.7170'}], 'svangerskap': False, 'hovedDiagnose': {'kode': 'X99', 'tekst': 'x', 'system': '2.16.578.1.12x.7170'}, 'yrkesskadeDato': None, 'annenFraversArsak': None}, 'skjermesForPasient': False, 'tiltakArbeidsplassen': None, 'syketilfelleStartDato': '2017-02-06', 'utdypendeOpplysninger': {}, 'meldingTilArbeidsgiver': None}, 'tlfPasient': None, 'mottattDato': '2x9:36', 'fellesformat': '', 'personNrLege': 'x', 'rulesetVersion': None}
b = { 'sykmelding': {'id': 'b', 'msgId': '', 'perioder': [{'fom': '2017-01-06', 'tom': '2017-03-05', 'gradert': 21}, {'fom': '2017-02-06', 'tom': '2017-03-05', 'gradert': 12}], 'meldingTilNAV': None, 'avsenderSystem': {'navn': 'System X', 'versjon': '3.9.1.3 (5015)'}, 'pasientAktoerId': '1000035002112', 'kontaktMedPasient': {'kontaktDato': None, 'begrunnelseIkkeKontakt': None}, 'behandletTidspunkt': '2017-02-05T23:00:00', 'medisinskVurdering': {'yrkesskade': False, 'biDiagnoser': [{'kode': 'R05', 'tekst': 'HOSTE', 'system': '2.16.578.1.12.4.1.1.7170'}], 'svangerskap': False, 'hovedDiagnose': {'kode': 'X99', 'tekst': 'x', 'system': '2.16.578.1.12x.7170'}, 'yrkesskadeDato': None, 'annenFraversArsak': None}, 'skjermesForPasient': False, 'tiltakArbeidsplassen': None, 'syketilfelleStartDato': '2017-02-06', 'utdypendeOpplysninger': {}, 'meldingTilArbeidsgiver': None}, 'tlfPasient': None, 'mottattDato': '2x9:36', 'fellesformat': '', 'personNrLege': 'x', 'rulesetVersion': None}
c = { 'sykmelding': {'id': 'c', 'msgId': '', 'perioder': [{'fom': '2017-01-06', 'tom': '2017-03-05', 'gradert': 19}, {'fom': '2017-02-06', 'tom': '2017-03-05', 'gradert': None}], 'meldingTilNAV': None, 'avsenderSystem': {'navn': 'System X', 'versjon': '3.9.1.3 (5015)'}, 'pasientAktoerId': '1000035002112', 'kontaktMedPasient': {'kontaktDato': None, 'begrunnelseIkkeKontakt': None}, 'behandletTidspunkt': '2017-02-05T23:00:00', 'medisinskVurdering': {'yrkesskade': False, 'biDiagnoser': [{'kode': 'R05', 'tekst': 'HOSTE', 'system': '2.16.578.1.12.4.1.1.7170'}], 'svangerskap': False, 'hovedDiagnose': {'kode': 'X99', 'tekst': 'x', 'system': '2.16.578.1.12x.7170'}, 'yrkesskadeDato': None, 'annenFraversArsak': None}, 'skjermesForPasient': False, 'tiltakArbeidsplassen': None, 'syketilfelleStartDato': '2017-02-06', 'utdypendeOpplysninger': {}, 'meldingTilArbeidsgiver': None}, 'tlfPasient': None, 'mottattDato': '2x9:36', 'fellesformat': '', 'personNrLege': 'x', 'rulesetVersion': None}

l.append(a)
l.append(b)
l.append(c)

data = pd.Series(l)

m = pd.json_normalize(data, record_path=['sykmelding','perioder'], meta=[['sykmelding','id']])

data0 = m.groupby('sykmelding.id').nth(0)
data1 = m.groupby('sykmelding.id').nth(1)
data2 = m.groupby('sykmelding.id').nth(2)

data0 = data0.reset_index()
data1 = data1.reset_index()
data2 = data2.reset_index()

d_tmp = pd.merge(data0, data1, on="sykmelding.id", how="left")
d_tmp = pd.merge(d_tmp, data2, on="sykmelding.id", how="left")


id = data.map(lambda x: x["sykmelding"]["id"])




n = pd.DataFrame({"fom_0": fom_0, "fom_1" : fom_1, "fom_2" : fom_2, "fom_3" : fom_3, "fom_4" : fom_4, "fom_len" : fom_len})
df = pd.merge(df, n, left_on="sykmelding_id", right_on=n["index"], how="left")




# p = a['sykmelding']



m = m.sort_values(by=['sykmelding.id', 'fom', 'tom'])



fom_0 = m.groupby('sykmelding.id').fom.nth(0)