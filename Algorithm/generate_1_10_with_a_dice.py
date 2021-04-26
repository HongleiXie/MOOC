# coding: utf-8
# Problem:
# https://math.stackexchange.com/questions/1314460/how-to-generate-a-random-number-between-1-and-10-with-a-six-sided-die
import random
import numpy as np

class Solution(object):
    def roll_dice(self):
        return random.randint(1,6)

    def map_to_1_10(self):
        step = 3
        res = []
        for i in range(1, 7):
            for j in range(1,7):
                if i == j:
                    continue
                a = [i,j]
                res.append(a)

        list_every_step = [res[i:i+step] for i in range(0, len(res), step)]
        res_dict = {}
        for i in range(10):
            mapping_keys  = [tuple(x) for x in list_every_step[i]]
            mapping_values = [i+1] * step
            d = dict(zip(mapping_keys, mapping_values))
            res_dict.update(d)
        return res_dict


    def generate_1_10_with_a_dice(self):
        """
        To generate a random number between 1 and 10 (uniformly) with an unbiased six-sided die
        How to do it?
        Roll the dice in pairs to generate pairs. Doubles don't count and are rerolled, e.g 1-1, 2-2, 3-3 etc...
        A roll of 1–2, 1–3, or 1–4 is the digit 0.
        A roll of 1–5, 1–6, or 2–1 is the digit 1.
        A roll of 2–3, 2–4, or 2–5 is the digit 2.
        And so on up to digit 9, which is a roll of 6–3, 6–4, or 6–5.
        :return: an int between 1 to 10
        """
        first_num = self.roll_dice()
        second_num = self.roll_dice()

        if first_num == second_num:
            while first_num == second_num:
                second_num = self.roll_dice()

        two_nums = (first_num, second_num)
        mapping_dict = self.map_to_1_10()
        random_num = mapping_dict.get(two_nums)
        return random_num

# verify if it's truly random
all_nums = []
N = 0
while N < 10000:
    generated_num = Solution().generate_1_10_with_a_dice()
    all_nums.extend([generated_num])
    N += 1

print({x: all_nums.count(x) for x in all_nums}) # ideally each digit would appear ~ 1000 times

# how many die throws on average to generate 1 digit
def expected_throws(N):
    assert N>=2
    res = 0
    for k in range(2, N+1):
        add_on = 5/6.0 * k * (1/np.power(6, k-2))
        res += add_on
        k += 1
    return res
# when N is big enough, it converges to ~2.2
