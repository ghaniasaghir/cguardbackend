from sqlalchemy.orm import sessionmaker
from main_fixed_all_normal_v2 import engine, UC_RiskDB  # import your engine and UC model

Session = sessionmaker(bind=engine)
session = Session()

for uc in session.query(UC_RiskDB).all():
    uc.risk_level = "Normal"
    uc.risk_percentage = 10
    session.add(uc)

session.commit()
session.close()
print("All 789 UCs updated to Normal successfully.")
