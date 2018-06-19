#numerical_stability

bignum=1 #one BILLLLLLION
smallnum=0.000001 #10^-6
one_million=1000000
billion = bignum

for i in range(0, one_million):
    bignum+=smallnum
    
    
bignum -= billion

print(bignum)