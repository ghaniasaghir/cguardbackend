from sqlalchemy.orm import sessionmaker
from main_fixed_all_normal_v2 import engine, UC_RiskDB  # Replace with your actual table/model

def set_all_ucs_normal():
    Session = sessionmaker(bind=engine)
    session = Session()

    # Update all 789 UCs
    all_ucs = session.query(UC_RiskDB).all()
    for uc in all_ucs:
        uc.risk_level = "Normal"
        uc.risk_percentage = 10  # safe Normal value
        session.add(uc)

    session.commit()
    session.close()
    print("All 789 UCs updated to Normal")

if __name__ == "__main__":
    set_all_ucs_normal()
