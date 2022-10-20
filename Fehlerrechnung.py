from Source import *

# ======= 2 ========
x = unc.ufloat(17.4, 0.3)
y = unc.ufloat(9.3, 0.7)

f = [x-y, 12*x + 3*y, 5*x*y, y**3/x**2, x**2 + 3*y**2, unp.arcsin(y/x), (3*x*y)**0.5, unp.log(y/x), x/y**2 + y/x**2, 2*(y/x)**0.5]

for i in range(len(f)):
    print(f"z{i+1} = {f[i]:.1uSL}")

print("\n")
# ======= 3 ========
# calculate velocity of object passing x1. x2 in time t
x1, x2 = unc.ufloat(5, 0.001), unc.ufloat(17, 0.001)
t = unc.ufloat(77283.5, 0.1)*10**(-6)
v = (x2-x1)/t
print(f"{v:.1uSL}")

