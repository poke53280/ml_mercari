

# Driver included.

import pandas as pd
import numpy as np
import datetime
from general.TimeAndDate import *

##############################################################################
#
#   TimeLineText
#

class TimeLineText:


    ##############################################################################
    #
    #   Constructor
    #

    def __init__(self, view0, view1, is_draw_small, is_draw_point_only, isVerbose, isClipEdge):
      self._view_0 = view0
      self._view_1 = view1
      self._rastersize = view1 - view0
      self._is_draw_small = is_draw_small
      self._is_draw_point_only = is_draw_point_only
      self._isVerbose = isVerbose
      self._isClipEdge = isClipEdge

    ##############################################################################
    #
    #   SetCustomRasterSize
    #

    def SetCustomRasterSize(self, rastersize):
        self._rastersize = rastersize

    ##############################################################################
    #
    #   GetPaddedFullRangeMinMax
    #

    def GetPaddedFullRangeMinMax(df_m):
        min_value = np.min(df_m.F1)
        max_value = np.max(df_m.Q)

        L = max_value - min_value

        rGrow = .1

        min_value = int(min_value - rGrow * L)
        max_value = int (max_value + rGrow * L)

        return (min_value, max_value)

    """c"""

    ##############################################################################
    #
    #   _GetCameraCoord()
    #
    # Returns location in camera space [0,1]
    #

    def _GetCameraCoord(self, r):
        x = (r - self._view_0) / (self._view_1 - self._view_0)
        return x

    ##############################################################################
    #
    #   _GetRasterLocationFromCameraCoord()
    #

    def _GetRasterLocationFromCameraCoord(self, c):

        pf = c * self._rastersize
        return pf
   
    ##############################################################################
    #
    #   GetInputFromRasterized()
    #
    #

    def GetInputFromRasterized(self, pn):
        x = pn/self._rastersize
        r = x * (self._view_1 - self._view_0) + self._view_0 
        return r

    ##############################################################################
    #
    #   GetRasterizedRangesFromSet()
    #
    #

    def GetRasterizedRangesFromSet(self, s):

        l = sorted(s)

        q = []

        for idx, x in enumerate(l):
            if idx == 0 or x != last +1:
                q.append(idx)

            last = x

        """c"""

        acMinMax = np.empty([len(q), 2], dtype=int)

        idx = 0

        while idx < len(q):
            lo_idx = q[idx]

            if idx +1 < len(q):
                hi_idx = q[idx +1] -1
            else:
                hi_idx = len(l) -1

            val_lo = l[lo_idx]
            val_hi = l[hi_idx] + 1 # For exclusive 

            acMinMax[idx,] = [val_lo, val_hi]

            idx = idx + 1

        return acMinMax



    ##############################################################################
    #
    #   GetInputRangesFromRasterizedRanges()
    #
    #

    def GetInputRangesFromRasterizedRanges(self, acMinMax):

        acMinMaxNew = np.empty(acMinMax.shape, dtype=int)

        for idx, row in enumerate(acMinMax):

            min_value = row[0]
            max_value = row[1]

            newMin = self.GetInputFromRasterized(min_value)
            newMax = self.GetInputFromRasterized(max_value)
        
            assert(newMax >= newMin)

            acMinMaxNew[idx, ] =  [newMin, newMax]

        return acMinMaxNew

    ##############################################################################
    #
    #   GetRangesFromSet()
    #
    #

    def GetRangesFromSet(self, s):
        return self.GetInputRangesFromRasterizedRanges(self.GetRasterizedRangesFromSet(s))

    ##############################################################################
    #
    #   ExpandRanges()
    #

    def ExpandRanges(self, acMinMax, nGrow):

        acMinMaxNew = np.empty(acMinMax.shape, dtype=int)

        for idx, row in enumerate(acMinMax):

            min_value = row[0]
            max_value = row[1]

            assert (max_value > min_value)

            newMin = min_value - nGrow
            newMax = max_value + nGrow

            assert (newMax > newMin)

            acMinMaxNew[idx, ] =  [newMin, newMax]

        return acMinMaxNew

    ##############################################################################
    #
    #   Wield()
    #
    #

    def Wield(self, s, wield_gap_min):

        r0 = self.GetRasterizedRangesFromSet(s)

        links = 0

        output = []

        start_min = r0[0][0]

        while links < len(r0) - 1:

            d = r0[links+ 1][0] - r0[links][1] + 1

            #print(f"{links} to {links +1 }, d = {d}")

            if d <= wield_gap_min:
                pass
                #print(f"===> Wield interval")

            else:
                #print(f"===> Don't wield - store and start collecting")
                end = r0[links][1]
                output.append( [start_min, end])
                start_min = r0[links +1][0]

            links = links + 1

        #print(f"Terminate run")
        output.append( [start_min, r0[-1][1] ])

        res = np.array(output)

        r_global = self.GetInputRangesFromRasterizedRanges(res)

        return self.GetRaster(r_global)

    ##############################################################################
    #
    #   ExpandRaster()
    #
    #

    def ExpandRaster(self, s, nGrow):
        r = self.GetRasterizedRangesFromSet(s)
        r_expanded = self.ExpandRanges(r, nGrow)
        r_global = self.GetInputRangesFromRasterizedRanges(r_expanded)

        return self.GetRaster(r_global)

    ##############################################################################
    #
    #   Rasterize()
    #
    #

    def Rasterize(self, r0, r1):

        ROUND_EPSILON = 0.001

        s = set()

        assert (r1 >= r0)

        c0 = self._GetCameraCoord(r0)
        c1 = self._GetCameraCoord(r1)

        assert (c1 >= c0)

        if c1 < 0:
            pass

        if c0 >= 1:
            pass

        pf0 = self._GetRasterLocationFromCameraCoord(c0)

        if (pf0 < 0):
            pf0 = 0

        pf1 = self._GetRasterLocationFromCameraCoord(c1)

        rSize = pf1 - pf0

        nSize = int (rSize + .5)

        if self._is_draw_small and nSize == 0:
            nSize = 1

        if nSize == 0:
            pass
        else:
            pn0 = int (pf0 + ROUND_EPSILON)

            if self._is_draw_point_only:
                pn1 = pn0
            else:
                pn1 = pn0 + nSize - 1

            if pn1 > self._rastersize -1:
                pn1 = self._rastersize -1

            r = range (pn0, pn1 + 1)

            s = s.union (r)

        return s

    ##############################################################################
    #
    #   RenderRasterized()
    #
    #

    def RenderRasterized(self, s, token):

         if len(s) == 0:
             return

         b = np.empty(self._rastersize, dtype='<U1')
         b.fill('-')
 
         for x in s:
            b[x] = token

         a_str = ''.join(str(x) for x in b)

         print(a_str)

    ##############################################################################
    #
    #   GetRaster()
    #
    #

    def GetRaster(self, acRange):
        acc_set = set()

        if len (acRange) == 0:
            pass
        else:
            for a in acRange:
                r0 = a[0]
                r1 = a[1]
                s = self.Rasterize(r0, r1)
                acc_set = acc_set.union(s)

        return acc_set

    ##############################################################################
    #
    #   Render()
    #
    #

    def Render(self, acRange, token):

        for a in acRange:
            r0 = a[0]
            r1 = a[1]
            s = self.Rasterize(r0, r1)
            self.RenderRasterized(s, token)

        acc_set = self.GetRaster(acRange)
        self.RenderRasterized(acc_set, token)


    ##############################################################################
    #
    #   DescribeScale()
    #
    #

    def DescribeScale(self):
        date_start = TimeAndDate_GetDateTimeFromEpochDays(int(self._view_0))
        date_end   = TimeAndDate_GetDateTimeFromEpochDays(int(self._view_1))

        zStart = TimeAndDate_GetDateTimeDescription(date_start)
        zEnd =  TimeAndDate_GetDateTimeDescription(date_end)

        rastersize = (self._view_1 - self._view_0) / self._rastersize

        if rastersize == 1:
            s = f"From {zStart} to {zEnd} at day resolution"
        else:
            s = f"From {zStart} to {zEnd} resolution {rastersize:.1f} days"

        return s

    ##############################################################################
    #
    #   expand_ranges()
    #
    #
    #  applicationPolicy: 0 : apply all left
    #  applicationPolicy: 1 : apply left and right as input
    #  applicationPolicy: 2 : apply all right


    def expand_ranges(self, acMinMax, acCount, applicationPolicy):

        assert (len(acCount) == len(acMinMax))

        acMinMaxNew = np.empty(acMinMax.shape, dtype=int)

        for idx, row in enumerate(acMinMax):

            min_value = row[0]
            max_value = row[1]

            countLeft = acCount[idx][0]
            countRight = acCount[idx][1]

            assert (countLeft >= 0 and countRight >= 0)
            assert (applicationPolicy == 0 or applicationPolicy == 1 or applicationPolicy == 2)

            if applicationPolicy == 0:
                applicationLeft = countLeft + countRight
                applicactionRight = 0

            elif applicationPolicy == 1:
                applicationLeft = countLeft
                applicactionRight = countRight

            elif applicationPolicy == 2:
                applicationLeft = 0
                applicactionRight = countLeft + countRight

            min_value = min_value + applicationLeft
            max_value = max_value - applicactionRight

            acMinMaxNew[idx] = [min_value, max_value]

        return acMinMaxNew


        s_added = s_filled.difference(s_original)

        if self._isVerbose:
            self.RenderRasterized(s_added, 'd')

        acMinMax = self.GetRasterizedRangesFromSet(s_filled)

        acCount = GetItemInRangesCount(s_added, acMinMax)

        return acCount


    ##############################################################################
    #
    #   CombineIntervals()
    #

    def CombineIntervals(self, r1, growConst):

        acc_set = self.GetRaster(r1)

        if len(acc_set) == 0:
            return np.empty((0,3), dtype=int)

        min_r = min (acc_set)
        max_r = max (acc_set)

        isNearEdgeBegin = (min_r <= (2 * growConst))
        isNearEdgeEnd   = (max_r >= (self._rastersize - 1 - 2 * growConst))

        if self._isVerbose:
            self.RenderRasterized(acc_set, 'o')

        s_final = self.Wield(acc_set, growConst * 2)

        if self._isVerbose:
            self.RenderRasterized(s_final, 'x')
    

        s_added = s_final.difference(acc_set)

        if self._isVerbose:
            self.RenderRasterized(s_added, 'd')

        acMinMaxRasterized = self.GetRasterizedRangesFromSet(s_final)

        acCount = GetItemInRangesCount(s_added, acMinMaxRasterized)

        assert (len (acMinMaxRasterized) == len (acCount))

        acMinMax = self.GetRangesFromSet(s_final)

        acResult = np.column_stack((acMinMax, acCount))

        if self._isClipEdge:
            if isNearEdgeBegin:
                acResult = acResult[1:]

            if isNearEdgeEnd:
                acResult = acResult[:-1]

        return acResult

    """c"""
    
    
    ##################################################################################
    #
    #     GetTarget
    #
    #
    # Todo: Provide documentation and example
    #

    def GetTarget(self, r_m, nGrow):

        r_m_excl = r_m.copy()
        r_m_excl[:, 1] += 1

        r_m_processed = self.CombineIntervals(r_m_excl, nGrow)

        # Go to end inclusive mode
        r_m_processed[:, 1] -= 1

        # Working on inclusive end values - i.e. both begin and end contained in interval.

        r_m_start = r_m[:, 0]
        r_m_end   = r_m[:, 1]

        group_idx = np.zeros(r_m.shape[0])

        for idx, p in enumerate(r_m_processed):
            a = p[0]
            b = p[1]

            # Fully inside range:
            m = (r_m_start >= a) & (r_m_end <= b)

            # Inside range a, b:
            # nInside = len(r_m[m])

            group_idx[m] = idx

            # print(f"#intervals in range [{a}, {b}]: {nInside}")


        isEmpty = len(r_m_processed) == 0

        result = {}
        result['nperiods'] = len(r_m_processed)

        if isEmpty:
            pass
        else:
            # Pick last period as target period

            target_idx = len (r_m_processed) - 1

            target_info = r_m_processed[-1]

            target_begin, target_end, target_stitch = target_info[0], target_info[1], target_info[2]

            result['begin'] = target_begin
            result['end'] = target_end
            result['stitch'] = target_stitch

            result['ids'] = []
   
            m = (group_idx == target_idx)

            interval_list = np.where(m)

            for x in interval_list[0]:
                val = r_m[x]
                result['ids'].append(x)

            # Collect non-target, earlier intervals:
            
            m = (group_idx != target_idx)

            checked_list = []

            interval_list = np.where(m)
            for x in interval_list[0]:
                val = r_m[x]
                if val[1] < target_begin:
                    checked_list.append(( val[0], val[1]))

            result['historic_intervals'] = checked_list

        return result

#############################################################################
#
#   IsItemInRange
#
#   Returns ranges [inclusive, exclusive>
#

def IsItemInRange(i, cMinMax):
    if i < cMinMax[0]:
        return False

    if i >= cMinMax[1]:
        return False

    return True

"""c"""

########################################################
#
#    GetIsInLowRange
#
# Asserts that item is in range. Returns true if item is in
# lower part of range
#
# Asserts that range is non empty
# 
# Returns True for items exactly in the middle of a range
# Returns True for item in length one range

def GetIsInLowRange(i, cMinMax):
    assert (IsItemInRange(i, cMinMax))

    nPixels = cMinMax[1] - cMinMax[0]
    assert (nPixels > 0)        # Non-empty range

    midPoint = cMinMax[0] + nPixels/2

    if (i < midPoint):
        return True

    else:
        return False

"""c"""

########################################################
#
#    GetItemInRangesCount
#

def GetItemInRangesCount(s, acMinMax):

    l = sorted(s)
    
    acCount = np.empty( acMinMax.shape[0], dtype=int)
    
    iCurrentItem = 0

    for iCurrentInterval, cMinMax in enumerate(acMinMax):
        iCurrentCount = 0
        
        while (iCurrentItem < len(l)) and (IsItemInRange(l[iCurrentItem], cMinMax)):

            iCurrentCount = iCurrentCount + 1
            iCurrentItem = iCurrentItem + 1

        acCount[iCurrentInterval] = iCurrentCount


    return acCount

"""c"""

########################################################
#
#    GetSetFromRanges
#

def GetSetFromRanges(acMinMax):

    s = set()

    for idx, row in enumerate(acMinMax):

        min_value = row[0]
        max_value = row[1]

        r = range (min_value, max_value)
        s = s.union(r)

    return s

"""c"""

########################################################
#
#    describe_intervals
#

def describe_intervals(acMinMax):
    for row in acMinMax:

        nDays = row[1] - row[0]
        assert (nDays > 0)

        print(f"Interval size = {nDays}")

"""c"""



########################################################
#
#    DisplayTargetInterval
#

def DisplayTargetInterval(df_m, start_target):
    
    data_count = 0

    idx = 0

    for x in start_target:

        isEmpty = x[0] == 0 and x[1] == 0

        if isEmpty:
            idx = idx + 1
            continue

        print(f"*********************************************** {idx} ****************************************")
        print(f" ")

        str = f"index = {idx}, interval [{x[0]}, {x[1]}]. L = {x[1] - x[0]}. L_comp = {x[2]}"
        print(str)

        t_start = x[0] - 10
        t_end   = x[1] + 10

        t = TimeLineText(t_start, t_end, True, False, False, False)

        r_m = get_F1_Q_ranges(df_m, idx)

        t.Render(r_m, 'x')

        AnalyzeTargetCondition(df_m, start_target, idx)

        data_count = data_count + 1
        idx = idx + 1

        print(f"  ")
        

    print(f"data/all = {data_count}/{len(start_target)}")

"""c"""    

########################################################
#
#    AnalyzeTargetCondition
#

def AnalyzeTargetCondition(df, out, idx):

    target_range_info = out[idx]

    t_start = target_range_info[0]
    t_end   = target_range_info[1]
    t_fill  = target_range_info[2]

    L_full = t_end - t_start
    L_adj = L_full - t_fill

    print(f"   INIT TIME: {t_start}")
    print(f"   LENGTH FULL: {L_full} days")
    print(f"   LENGTH ADJ : {L_adj} days")

    m = df.IDX == idx
    pt = df[m]

    m = (pt.F1 >= t_start) 
    pt = pt[m]

    m = (pt.Q < t_end)
    pt = pt[m]

    pt = pt.sort_values(by=['Q'])

    if len(pt) == 0:
        print(f"WARNING: NO DATA")
    else:
        q_first = pt.Q.data[0]
        time_into_state = q_first - t_start 
        print(f"   First data concluded {time_into_state} day(s) into target state:")
        print (pt[0:1])


"""c"""



##############################################################################
#
#        TimeLineTool_GetProximityGroups1D
#
#   Returns array with indices for each input element in n
#
#   TimeLineTool_GetProximityGroups1D(np.array([3,4,5,11]), 3)
#   => array([0, 0, 0, 1])
#

def TimeLineTool_GetProximityGroups1D(n, threshold):

    assert (threshold >= 0)

    nID = 0

    e = np.zeros(len (n), dtype = np.int)

    last_value = -1     # impossible value

    for idx, x in np.ndenumerate(n):
        if last_value == -1:
            pass
        else:       
            diff = x - last_value

            if diff <= threshold:
               pass
            else:
                nID = nID + 1

        e[idx] = nID
    
        last_value = x

    assert(len(e)== len(n))

    return e

"""c"""


##############################################################################
#
#        TimeLineTool_cluster_events
#

def TimeLineTool_cluster_events(acData):

    n = len (acData)

    min_value = acData.min()
    max_value = acData.max()

    print(f" {n} event(s). min: {min_value} max: {max_value}. length = {1 + max_value - min_value}")

    lcNCluster = []
    lcClusterChange = []

    e = TimeLineTool_GetProximityGroups1D(acData, 0)
    nClusters = e.max() + 1
    lcNCluster.append(nClusters)
    lcClusterChange.append(0)

    fClusterSize = n/nClusters

    iThreshold = 1

    while nClusters > 1:

        e = TimeLineTool_GetProximityGroups1D(acData, iThreshold)
        nClusters = e.max() + 1

        if nClusters < lcNCluster[-1]:
            lcNCluster.append(nClusters)
            lcClusterChange.append(iThreshold)

            fClusterSize = n/nClusters

        iThreshold = iThreshold + 1

    d = dict (zip (lcNCluster, lcClusterChange))

    return d

"""c"""


##############################################################################
#
#   TimeLineTool_analyse_user_code
#

def TimeLineTool_analyse_user_code(df, idx):

    print(f"Analyzing user code {idx}...")
    m = df.user_code == idx
    q = df[m]

    print(f"#Values: {len(q)}")

    print(q)

    acData = q.time.values

    d = TimeLineTool_cluster_events(acData)

    return d

"""c"""

##############################################################################
#
#   TimeLineTool_GetGroups
#

def TimeLineTool_GetGroups(acData, anGroup):

    assert (anGroup.min() == 0)

    nGroups = anGroup.max() + 1

    l = []
    for u in range(0, nGroups):
        begin = np.searchsorted(anGroup, u)
        end   = np.searchsorted(anGroup, u+1)

        first_element = acData[begin]
        last_element = acData[end - 1]

        length = last_element - first_element + 1

        center = first_element + length/2

        element_count = end -1 - begin + 1

        density = element_count / length

        l.append( (center, length/2, element_count))

        # print(f"begin = {begin}, end = {end}, first = {first_element}, last = {last_element}, density = {density}")

    return l

"""c"""

##############################################################################
#
#   TimeLineTool_Analyze_Cluster
#

def TimeLineTool_Analyze_Cluster(acData, nProximityValue):

    anGroup = tl.TimeLineTool_GetProximityGroups1D(acData, nProximityValue)

    l = TimeLineTool_GetGroups(acData, anGroup)

    score = 0

    for x in l:
        group_range = x[1] * 2
        element_count = x[2]
        density = element_count / group_range
        # print(f"range = {group_range}, element_count = {element_count}, density = {density} ")
        score = score + density * element_count

    return score
   
"""c"""

##############################################################################
#
#   TimeLineTool_GetOptimalGroupSize
#

def TimeLineTool_GetOptimalGroupSize(acData, isGraph, max_grow):

    r = range(acData.max())

    acRandomData = np.random.choice(r, len (acData))
    acRandomData = np.sort(acRandomData)

    global_density = len(acData) / (acData.max() - acData.min() + 1)
    print(f"global_density {global_density}")

    global_density_random = len(acRandomData) / (acRandomData.max() - acRandomData.min() + 1)
    print(f"global_density_random {global_density_random}")

    xRange = max_grow   # Largest foreseen grouping
    lcProx = np.array (range(xRange))
    lcProx = lcProx       

    y_real = []
    y_random = []
    y_diff = []

    for n, x_value in enumerate(lcProx):
        score_real = Analyze_Cluster(acData, x_value)
        score_rand = Analyze_Cluster(acRandomData, x_value)

        y_real.append(score_real)
        y_random.append(score_rand)
        y_diff.append(score_real - score_rand)

        # print(f"{n/xRange}")

    """c"""

    acDiff = np.array(y_diff)

    an = np.argsort(acDiff)

    maxIndex = an[-1]  # xxx add offset
    acDiff[maxIndex]

    print(f"attr {maxIndex} diff {acDiff[maxIndex]}")
    
    if isGraph:
        plt.plot(lcProx, y_real)
        plt.plot(lcProx, y_random)
        plt.plot(lcProx, y_diff)
        plt.show()

    return maxIndex

"""c"""    
