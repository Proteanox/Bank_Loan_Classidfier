import pickle
import streamlit as st

# loading the trained model
pickle_in = open('classifier.pkl', 'rb')
model = pickle.load(pickle_in)


@st.cache()
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):
    # Pre-processing user input
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1

    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1

    if Credit_History == "Unclear Debts":
        Credit_History = 0
    else:
        Credit_History = 1

    LoanAmount = LoanAmount / 1000

    # Making predictions
    predictions = model.predict(
        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])

    if predictions == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred


def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:pink;padding:15px"> 
    <h1 style ="color:white;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    Gender = st.selectbox('Gender', ("Male", "Female"))
    Married = st.selectbox('Marital Status', ("Unmarried", "Married"))
    ApplicantIncome = st.number_input("Applicants monthly income")
    LoanAmount = st.number_input("Total loan amount")
    Credit_History = st.selectbox('Credit_History', ("Unclear Debts", "No Unclear Debts"))
    result = " "

    if st.button("Predict"):
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History)
        st.success('Your loan is {}'.format(result))
        print(LoanAmount)


if __name__ == '__main__':
    main()

