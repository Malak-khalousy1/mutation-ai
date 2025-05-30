
import streamlit as st
import pandas as pd
import pickle

# تحميل النموذج والمشفّرات
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    le_gene, le_mutation, le_change = pickle.load(f)

# واجهة التطبيق
st.set_page_config(page_title="تحليل الطفرات الجينية", layout="centered")
st.title("🔬 نظام تحليل الطفرات الجينية باستخدام الذكاء الاصطناعي")

st.markdown("هذا التطبيق يساعد في التنبؤ بما إذا كانت طفرة جينية معينة قد تكون مسببة للسرطان، باستخدام نموذج ذكاء اصطناعي مدرّب.")

st.header("🧬 أدخل بيانات الطفرة:")

gene = st.selectbox("اختر الجين:", le_gene.classes_)
mutation_type = st.selectbox("نوع الطفرة:", le_mutation.classes_)
position = st.number_input("الموقع داخل الجين:", min_value=1, max_value=2000, step=1)
amino_change = st.selectbox("تغير الحمض الأميني:", le_change.classes_)

if st.button("🔎 تحليل الطفرة"):
    try:
        # تحويل القيم
        gene_enc = le_gene.transform([gene])[0]
        mut_enc = le_mutation.transform([mutation_type])[0]
        change_enc = le_change.transform([amino_change])[0]

        # التنبؤ
        features = [[gene_enc, mut_enc, position, change_enc]]
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        # عرض النتيجة
        st.subheader("🧾 النتيجة:")
        if prediction == 1:
            st.success(f"🔴 هذه الطفرة يُحتمل أن تكون مسببة للسرطان بنسبة {proba*100:.2f}%")
        else:
            st.info(f"🟢 هذه الطفرة غير مرجّح أن تكون مسببة للسرطان. نسبة الاحتمال: {proba*100:.2f}%")

        # عرض التفسير المبسط
        st.markdown("**تفسير مبسّط:**")
        st.markdown("- تم تحليل نوع الطفرة وموقعها داخل الجين.")
        st.markdown("- تم مقارنة الطفرة ببيانات سابقة لمعرفة مدى ارتباطها بالأورام.")

    except Exception as e:
        st.error("حدث خطأ أثناء التحليل: " + str(e))
