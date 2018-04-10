
####################### ID CONVERTERS #################################


class IDConverter:
    dictFK_TO_FID =  {}
    dictFID_TO_FK = {}
    dictA_TO_FID = {}
    dictFID_TO_A = {}

    def __init__(self, p):

        # Remove entries with invalid FK
        m = (p.FK != -1)
        p = p[m]

        #Check length of FID string
        q = p.FID.apply(lambda x: len(x))

        m = (q == 11)
        del q

        m.value_counts()
        #All FIDs at len 11

        l_fid = p.FID.tolist()
        l_fk = p.FK.tolist()

        self.dictFK_TO_FID = dict(zip (l_fk, l_fid))
        self.dictFID_TO_FK = dict(zip (l_fid, l_fk))

        s_fid = set(l_fid)
        s_fk  = set(l_fk)

        #More FIDs than FKs.
        print(f"More FIDs than FKs: {(len (s_fid) - len (s_fk))}")

        # Several entries with invalid A.
        m = (p.A >= 0)
        m.value_counts()

        # Remove them
        p = p[m]

        len (p)

        l_fid = p.FID.tolist()
        l_a   = p.A.tolist()

        self.dictA_TO_FID  = dict(zip (l_a, l_fid))
        self.dictFID_TO_A  = dict(zip (l_fid, l_a))

        len (self.dictA_TO_FID)
        len (self.dictFID_TO_A)

        print(f"A very few more FIDS than A: {(len (self.dictFID_TO_A) - len (self.dictA_TO_FID))}")

    def GetFID_FROM_FK(self, fk_person1):
        try:
            fid = self.dictFK_TO_FID[fk_person1]
            assert (len(fid) == 11)
            return fid
        except:
            return "None"

    def GetFK_FROM_FID(self, fid):
        try:
            return self.dictFID_TO_FK[fid]
        except:
            return "None"

    def GetConversionErrorMask_FID_FROM_FK(self, sFK):
        q = sFK.apply(lambda x: not self.GetFID_FROM_FK(x) == "None")
        q.value_counts()

        return q


    def Create_FID_FROM_FK(self, sFK):
        q = sFK.apply(lambda x: self.GetFID_FROM_FK(x))

        return q

    def Add_FID_FROM_FK_Delete_Missing(self, df):
        q = self.GetConversionErrorMask_FID_FROM_FK(df.FK)
        q_stat = GetTrueAndFalse(q)

        print(f"Mapping: {q_stat['True']}, Deleting: {q_stat['False']}")

        # Deleting missing from dataset:
        
        df = df[q]

        s = self.Create_FID_FROM_FK(df.FK)

        df = df.assign(FID=s.values)

        return df

"""c"""

p = d_all['pmap']

def preprocess_pmap(p):

    p.columns = ['FID', 'FK', 'A']

    p.isnull().sum()

    s = p.FID.apply(lambda x: get_birth_year_from_fid(x))

    p = p.assign(BIRTH = s)

    m = s == -1

    m.value_counts()

    p = p[~m]

    epoch_day = datetime.date(1970, 1, 1)

    s = p.BIRTH.apply(lambda x: get_random_epoch_birth_day_from_birth_year(epoch_day, x))

    p = p.assign(RND_BIRTH_EPOCH = s)

    p['SAKSKODE'] = 'AKTIV'

    p['FRA'] = s + (18 * 365)
    p['TIL'] = s + (67 * 365)

    p.drop(['BIRTH'], inplace = True, axis = 1)
    p.drop(['RND_BIRTH_EPOCH'], inplace = True, axis = 1)

    return p

"""c"""

id_converter = IDConverter(p)


