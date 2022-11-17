from Source import *
import mpl_interactions.ipyplot as iplt
from mpl_interactions import interactive_plot

# %%
# load data
data = np.loadtxt("/home/taco/Documents/Grundpraktikum/data/Versuch4_2.csv", delimiter=",", skiprows=1)
# create dataframe
df = pd.DataFrame(data, columns=["t", "I_sound"])

df["I_light_err"] = 0.001
df["I_sound_err"] = 0.001
rate = 4800
# 28.8, 29.65
L = unc.ufloat(0.817 + 0.6*0.426, 0.001, "L")
# L = unc.ufloat(0.422 + 0.6*0.31, 0.001, "L")

# plot data
plt.scatter(df["t"], df["I_sound"], s=0.2)
plt.xlabel("Zeit / s")
plt.ylabel("Intensität / a.u.")
plt.title("Intensität des Schalls")
plt.show()
# print(df["t"][12*rate:int(12.2*rate)])

# compute fft
# for i in [12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13]:
#     fft = np.fft.fft(df["I_sound"][12*rate:int(i*rate)])
#     freq = np.fft.fftfreq(len(fft), 1/rate)
#     plt.plot(freq, np.abs(fft), label=f"{i}s")

fft = np.fft.fft(df["I_sound"][int(12.8*rate):int((12.8+0.1)*rate)])
# print(fft.shape)
# print(np.abs(fft).shape)
freq = np.fft.fftfreq(len(df["t"][int(12.8*rate):int((12.8 + 0.1)*rate)]), d=1/rate)
# d = np.abs(fft)
# print(freq.shape)
fft1 = np.fft.fft(df["I_sound"][int(15.5*rate):int((15.5 + 0.1)*rate)])
freq1 = np.fft.fftfreq(len(df["t"][int(15.5*rate):int((15.5 + 0.1)*rate)]), d=1/rate)
# print(freq1)
# fft2 = np.fft.fft(df["I_sound"][12*rate:int(12.5*rate)])
# freq2 = np.fft.fftfreq(len(df["t"][12*rate:int(12.5*rate)]), d=1/rate)
# # print(freq2==freq1)
# plt.plot(freq, np.abs(fft), label="fft1")
# plt.plot(freq1, np.abs(fft1), label="fft2")
plt.legend()
# plt.plot(freq2, np.abs(fft2))
# create interavtive plot with slider for fft interval
# def x(t):
#     t = int(t)
#     return np.fft.fftfreq(len(df["t"][int(t) * rate:int((t + 0.1) * rate)]), d=1/rate)
#
x = np.concatenate((freq[25:], freq[:25]))
# x = np.linspace(-2400, 2400, 50)
def y(x, t):
    t = int(t)
    arr = np.abs(np.fft.fft(df["I_sound"][t:t+50]))
    arr[0] = 0
    # swap last half of array to the front
    return np.concatenate((arr[25:], arr[:25]))
    # return arr


# f1, f2 = 4800 - 2280, 4800 - 2180
f1, f2 = 1350, 1550
print((f2-f1)*2*L)
# plt.plot(x, y(x, 30001))
# y = np.abs(np.fft.fft(df["I_sound"][0:50]))
# print(y)
# controls = iplt.plot(x, y, t=(1, 50, 50), label="f1")
# iplt.plot(x, y, controls=controls, label="f2")
# iplt.plot(x, y, t=np.arange(1, 250000, 1, dtype=float), label="f1", ylim=(0, 25))
# fig, ax, sliders = interactive_plot(y, x=x, t=np.arange(1, 250000, 1, dtype=float), label="f1", ylim=(0, 25), xlim=(-2400, 2400))
# plt.xlabel("Frequenz / Hz")
# plt.ylabel("Amplitude / a.u.")
# plt.title("FFT des Schalls")
# plt.ylim(0, 25)
plt.show()

# plot_ft(2500, df["t"][12*rate:int(12.5*rate)].to_numpy(), df["I_sound"][12*rate:int(12.5*rate)].to_numpy())
# %%
def here(t):
    return t


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(df["t"], df["I_sound"], s=0.2)
# iplt.scatter(here, 3, t=np.arange(1, 250000, 5000, dtype=float), color="red")
iplt.plot(x, y, t=np.arange(1, 250000, 5000, dtype=float), label="f1")
plt.show()
# %%
x = np.linspace(1, 100, 100000)
y = sine(x, 1, 1, 0, 0) + sine(x, 2, 3, 0, 0)
d = {"x": x, "y": y}
df2 = pd.DataFrame(d)
print("x: ", x)
# plt.plot(x, y, color="magenta")
# plot_ft(10, x, y)
plot_ft(10, df2["x"].to_numpy(), df2["y"].to_numpy())
plt.show()

# %%
x = np.linspace(0, np.pi, 100)
tau = np.linspace(0.5, 10, 100)

def f1(x, tau, beta):
    return np.sin(x * tau) * x * beta
def f2(x, tau, beta):
    return np.sin(x * beta) * x * tau


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(df["t"], df["I_sound"], s=0.2)
controls = iplt.plot(x, f1, tau=tau, beta=(1, 10, 100), label="f1")
iplt.plot(x, f2, controls=controls, label="f2")
_ = plt.legend()
plt.show()
