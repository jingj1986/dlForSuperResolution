#!/usr/bin/python

#Author: Guojun Jin <>
#Date:   2016-10-13
#Desc:   The parent Class for image quota
#        1. Support different configures about when end 
#        2. Save the quotas at each time
#
## When Stop
#  END_BAD : end immediately when quota be bad, config NONE
#  END_BAD_TIME: end when quota bad some times, config: bad times to stop
#  END_UP_SLOW:  end when quota updating very slowly, config: times and converge range
#                       end when in X times, quota converges less than range
#
# Note:  we supported that, with the image better, the quota be bigger!

class Quota:
    def __init__(self, end_type, conf_time=0, conf_range=1.0):
        self.name = "BasicQuota"
        self.end_type = end_type
        self.conf_time = conf_time
        self.conf_range = 1.0*conf_range

        self.quotas = []
        self.bad_time = 0      #mem the bading times 
         
    def need_end(self):
        if self.end_type == "END_BAD":
            if len(self.quotas) >= 2:
                if self.quotas[-1] < self.quotas[-2]:  # 
                    return True
            return False

        elif self.end_type == "END_BAD_TIME":
            if len(self.quotas) >= 2:
                if self.quotas[-1] <= self.quotas[-2]:
                    self.bad_time = self.bad_time + 1
                else:
                    self.bad_time = 0
            if self.bad_time >= self.conf_time:
                return True
            else:
                return False
            
        elif self.end_type == "END_UP_SLOW":
            if len(self.quotas) >= self.conf_time:
                tmp_list = self.quotas[-self.conf_time:]
                if (max(tmp_list) - min(tmp_list))*1.0 < self.conf_range:
                    return True
            return False

        ## bad config, do not end the process by this quota
        return False

    def get_name(self):
        return self.name

    ## Used for top function
    def get_quota(self, img1 = None, img2 = None, *tupleArg):
        quota = self._calculate(img1, img2, tupleArg)
        self.quotas.append(quota)
        return quota

    ## SubClass just need to re-implement this function
    def _calculate(self, img1 = None, img2 = None, *tupleArg):
        return 0.1

# TEST
if __name__ == "__main__":
    obj = Quota('END_BAD')
    quota = obj.get_quota()
    end = obj.need_end()

    print quota, end
