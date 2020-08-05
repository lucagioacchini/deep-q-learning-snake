# =============
# PYGAME PARAMS
# =============
SPEED = 1000

H = 620 # Testing screen height
#H = 320 # Training screen height
W = H # Screen width


# ======================
# DEEP Q-LEARNING PARAMS
# ======================
GAMMA = .9  # Discounting factor

EPISODE = 1000  # Training episodes

STATE_L = 11  # State vector size

L1 = 150  # Number of neurons of the 1st layer
L2 = 150  # Number of neurons of the 2nd layer
L3 = 150  # Number of neurons of the 3rd layer
LR = 0.001  # Learning rate

# ==============================
# EPSILON GREEDY STRATEGY PARAMS
# ==============================
START = 1
END = .03
DECAY = .001

# ========
# REWARDS
# ========
DIED = -10.0
ATE = 10.0

# ====================
# REPLAY MEMORY PARAMS
# ====================
BATCH = 5000
CAPACITY = 1000000

#===================================
# MAXIMUM NUMBER OF STATISTICS SHOWN
#===================================
STAT_LIM = 100
