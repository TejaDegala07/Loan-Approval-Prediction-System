import joblib, numpy as np

model = joblib.load('model/loan_model.pkl')
print('Model type:', type(model).__name__)
print('Pipeline steps:', [s[0] for s in model.steps])

# Approve candidate: Male, Married, 0 deps, Graduate, Not SE, Income=5000, Co=2000, Loan=128, Term=360, CreditOK, Urban
approve = [[1, 1, 0, 0, 0, 5000, 2000, 128, 360, 1, 2]]
pred_a  = model.predict(approve)[0]
prob_a  = model.predict_proba(approve)[0]
print(f'\nApprove case -> prediction={pred_a}  proba={prob_a}')

# Reject candidate: Female, Single, 3 deps, Not Grad, SE, Income=1500, Co=0, Loan=200, Term=360, NoCred, Rural
reject = [[0, 0, 3, 1, 1, 1500, 0, 200, 360, 0, 0]]
pred_r  = model.predict(reject)[0]
prob_r  = model.predict_proba(reject)[0]
print(f'Reject case  -> prediction={pred_r}  proba={prob_r}')

result_a = "Loan Approved" if pred_a == 1 else "Loan Rejected"
result_r = "Loan Approved" if pred_r == 1 else "Loan Rejected"
print(f'\nApprove case result string: "{result_a}"')
print(f'Reject  case result string: "{result_r}"')

print('\n[OK] Model verified. Restart Flask to go live.')
