# Utility function

# Takes a number A and B and returns the middle value of both by the factor t
# For example if A is 10, B is 20 and t is 0.5 it would return 15
def lerp(A, B, t):
    return A+(B-A)*t