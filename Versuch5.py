import numpy as np

from Source import *
# %%
R1, R2 = unc.ufloat(10000, 10, "R1"), unc.ufloat(10000, 10, "R2")
R12 = R1 + R2
Vbatt1, Violab1 = unc.ufloat(0, 0), unc.ufloat(0, 0)
for i in range(1, 3):
    # load data
    df = pd.read_csv(f"data/Versuch5_{i}.csv")
    # rename columns
    df.columns = ["t", "V"]
    rate = get_polling_rate(df)

    # calculate mean and std of first 5 seconds
    mean, std = df["V"][0:5*rate].mean(), df["V"][0:5*rate].std()
    if i == 1:
        Vbatt1 = unc.ufloat(mean, std, "Vbatt1")
    else:
        Violab1 = unc.ufloat(mean, std, "Violab1")


Vbatt = (R1/R2 + 1)*Vbatt1
Violab = (R1/R2 + 1)*Violab1
print(f"Vbatt = {Vbatt:.1uS} V: Violab = {Violab:.1uS} V")
#
contributions(Vbatt)
# contributions(Violab)

# %%
R1, R2 = unc.ufloat(10000, 10, "R1"), unc.ufloat(4700, 4.7, "R2")
Shunt = unc.ufloat(1, 0.05, "Shunt")
for i in range(1, 3):
    # load data
    df = pd.read_csv(f"data/Versuch5_{i+2}.csv")
    # rename columns
    df.columns = ["t", "V"]
    rate = get_polling_rate(df)

    # calculate mean and std of first and last  4.5 seconds
    mean1, std1 = df["V"][0:int(4.5*rate)].mean(), df["V"][0:int(4.5*rate)].std()
    mean2, std2 = df["V"][int(-4.5*rate):].mean(), df["V"][int(-4.5*rate):].std()
    V1 = unc.ufloat(mean1, std1, "V1")
    V2 = unc.ufloat(mean2, std2, "V2")
    print(f"V1 = {V1:.2uS} V: V2 = {V2:.1uS} V")
    V = (np.abs(V1) + np.abs(V2))/2
    print(f"V = {V:.1uS} V")
    I = np.abs(V/Shunt)
    # print(f"offset{i}: {offset:.1uS} V")
    if i == 1:
        Iideal = Violab/R1*1000
        print(f"relative err: {Violab / R1 - Violab / (Shunt + R1):.1uS}")
    else:
        Iideal = Violab/R2*1000
        print(f"relative err: {Violab / R2 - Violab / (Shunt + R2):.1uS}")
    print(f"I{i}: {I:.1uS} mA: Iideal{i}: {Iideal:.2uS} mA")


# %%
R1, R2, R3, R4 = unc.ufloat(4700, 4.7, "R1"), unc.ufloat(10000, 10, "R2"), unc.ufloat(4700, 4.7, "R3"), unc.ufloat(10000, 10, "R4")
R34 = R3 + R4
R234 = R34*R2/(R34 + R2)
Rges = R234 + R1
Shunt = unc.ufloat(1, 0.05, "Shunt")
for i in range(1, 4):
    # load data
    df = pd.read_csv(f"data/Versuch5_{i+4}.csv")
    # rename columns
    df.columns = ["t", "V"]
    rate = get_polling_rate(df)

    # calculate mean and std of first and last  4.5 seconds
    mean1, std1 = df["V"][0:int(4.5*rate)].mean(), df["V"][0:int(4.5*rate)].std()
    mean2, std2 = df["V"][int(-4.5*rate):].mean(), df["V"][int(-4.5*rate):].std()
    V1 = unc.ufloat(mean1, std1, "V1")
    V2 = unc.ufloat(mean2, std2, "V2")
    V = (np.abs(V1) + np.abs(V2))/2
    print(f"A{i}:")
    if i == 1:
        I1 = np.abs(V / Shunt)
        Iideal1 = Violab/Rges*1000
        print(f"V1= {V + R1 * I1 / 1000:.1uS}V: Videal = {R1 * Iideal1 / 1000:.1uS} V")
        print(f"I = {I1:.1uS} mA: Iideal = {Iideal1:.1uS} mA")
    elif i == 2:
        V2 = V
        I2 = np.abs(V / Shunt)
        Iideal2 = (Violab - R1*Iideal1/1000)/R2*1000
        print(f"V2 = {(Shunt+R2)*I2/1000:.1uS}V: Videal = {R2*Iideal2/1000:.1uS} V")
        print(f"I = {I2:.1uS} mA: Iideal = {Iideal2:.1uS} mA")
    else:
        I3 = np.abs(V / Shunt)
        Iideal3 = (R2*Iideal2/1000)/R34*1000
        print(f"I = {I3:.1uS} mA: Iideal = {Iideal3:.1uS} mA")

U2 = Violab-I1*R1/1000
U1 = Violab - U2
U3 = I3*R3/1000
U4 = U2 - U3

U2ideal = Iideal2*R2/1000
U3ideal = Iideal3*R3/1000
U4ideal = Iideal3*R4/1000
U1ideal = Iideal1*R1/1000
print(f"U1 = {U1:.1uS} V: U1ideal = {U1ideal:.1uS} V")
print(f"U2 = {U2:.1uS} V: U2ideal = {U2ideal:.1uS} V")
print(f"U3 = {U3:.1uS} V: U3ideal = {U3ideal:.1uS} V")
print(f"U4 = {U4:.1uS} V: U4ideal = {U4ideal:.1uS} V")

# %%
# load data
df = pd.read_csv("data/Versuch5_8.csv")
# rename columns
df.columns = ["t", "V"]
rate = get_polling_rate(df)

end, endstd = df["V"][0:int(4.5*rate)].mean(), df["V"][0:int(4.5*rate)].std()
mid, midstd = df["V"][int(-4.5*rate):].mean(), df["V"][int(-4.5*rate):].std()
mid, end = unc.ufloat(mid, midstd, "mid"), unc.ufloat(end, endstd, "end")

U1 = end
U2 = Violab - end
U3 = mid - end
U4 = Violab - mid
print(f"U1 = {U1:.1uS} V")
print(f"U2 = {U2:.1uS} V")
print(f"U3 = {U3:.1uS} V")
print(f"U4 = {U4:.1uS} V")

# %%
a = np.array([0, 5, 9.5, 14.5, 20.7, 24.4, 30.5, 35.5, 40, 45, 50, -1])
b = np.array([0, 5, 10, 15, 18, 23, 30, 35, 41, 46, 53, -1])
c = np.array([0, 5, 10, 15, 19, 24, 32, 37, 45, 50, 58, -1])
# create 3x3 array
V = unp.umatrix(np.empty((3, 3)), np.empty((3, 3)))
for (i, j) in zip(range(1, 4), [a, b, c]):
    # load data
    df = pd.read_csv(f"data/Versuch5_{i+8}.csv")
    # rename columns
    df.columns = ["t", "V"]
    rate = get_polling_rate(df)
    # print(len(j))
    for index in range(len(j)//4):
        # print(index)
        mean1, meanstd1 = df["V"][int(j[index*4]*rate):int(j[index*4+1]*rate)].mean(), df["V"][int(j[index*4]*rate):int(j[index*4+1]*rate)].std()
        mean2, meandstd2 = df["V"][int(j[index*4+2]*rate):int(j[index*4+3]*rate)].mean(), df["V"][int(j[index*4+2]*rate):int(j[index*4+3]*rate)].std()
        V1 = unc.ufloat(mean1, meanstd1, "V1")
        V2 = unc.ufloat(mean2, meandstd2, "V2")
        V[i-1, index] = (np.abs(V1) + np.abs(V2))/2
        # print(V[i-1, index])

U1 = (V[0, 0] + V[0, 1] + V[0, 2])/3
U2 = (V[1, 0] + V[1, 1] + V[1, 2])/3
U3 = (V[2, 0] + V[2, 1] + V[2, 2])/3
print(f"U1 = {U1:.1uS} V")
print(f"U2 = {U2:.1uS} V")
print(f"U3 = {U3:.1uS} V")
