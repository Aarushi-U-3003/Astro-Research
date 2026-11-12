import lalsimulation as lalsim

# Iterate through the entire internal enum list to print all registered approximants
for i in range(lalsim.NumApproximants):
    try:
        name = lalsim.GetStringFromApproximant(i)
        if name and name != "NumApproximants":
            print(f"ID {i}: {name}")
    except Exception:
        pass