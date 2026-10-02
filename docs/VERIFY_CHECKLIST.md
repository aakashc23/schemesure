# Verification checklist

This file exists so the scheme data can be **spot-checked by a human**. Everything in SchemeSure is built on these 15 documents, so if a figure here is wrong, every answer about that scheme is wrong too.

**How to use it:** open a scheme's source URL, find the facts listed below, and tick the box if they match. Each eligibility rule is shown next to the exact official sentence it was derived from, so you are comparing like with like rather than re-interpreting the page.

> All data comes from the official myScheme portal (`myscheme.gov.in`, run by Digital India Corporation, Ministry of Electronics & IT), read through the public JSON API that the portal's own pages use. Raw API responses are kept in `data/raw/` so any fact can be traced back to the response it came from.

*Generated 2026-10-02 by `scripts/make_verify_checklist.py`*

---

## At a glance

| # | Scheme | Ministry | Verified on |
|---|--------|----------|-------------|
| 1 | [Atal Pension Yojana](https://www.myscheme.gov.in/schemes/apy) | Ministry Of Finance | 2026-10-02 |
| 2 | [Kisan Credit Card](https://www.myscheme.gov.in/schemes/kcc) | Ministry Of Agriculture and Farmers Welfare | 2026-10-02 |
| 3 | [National Means-cum-Merit Scholarship Scheme](https://www.myscheme.gov.in/schemes/nmmss) | Ministry of Education | 2026-10-02 |
| 4 | [Pradhan Mantri Fasal Bima Yojna (PMFBY)](https://www.myscheme.gov.in/schemes/pmfby) | Ministry Of Agriculture and Farmers Welfare | 2026-10-02 |
| 5 | [Pradhan Mantri Kisan Samman Nidhi](https://www.myscheme.gov.in/schemes/pm-kisan) | Ministry Of Agriculture and Farmers Welfare | 2026-10-02 |
| 6 | [Pradhan Mantri Matru Vandana Yojana](https://www.myscheme.gov.in/schemes/pmmvy) | Ministry of Women and Child Development | 2026-10-02 |
| 7 | [Pradhan Mantri Mudra Yojana](https://www.myscheme.gov.in/schemes/pmmy) | Ministry Of Finance | 2026-10-02 |
| 8 | [PM Street Vendor’s AtmaNirbhar Nidhi (PM SVANidhi)](https://www.myscheme.gov.in/schemes/pm-svanidhi) | Ministry Of Housing & Urban Affairs | 2026-10-02 |
| 9 | [Pradhan Mantri Ujjwala Yojana](https://www.myscheme.gov.in/schemes/pmuy) | Ministry Of Petroleum and Natural Gas | 2026-10-02 |
| 10 | [PM Vishwakarma](https://www.myscheme.gov.in/schemes/pmv) | Ministry Of Micro, Small and Medium Enterprises | 2026-10-02 |
| 11 | [Pradhan Mantri Awas Yojana - Urban](https://www.myscheme.gov.in/schemes/pmay-u) | Ministry Of Housing & Urban Affairs | 2026-10-02 |
| 12 | [Pradhan Mantri Jeevan Jyoti Bima Yojana](https://www.myscheme.gov.in/schemes/pmjjby) | Ministry Of Finance | 2026-10-02 |
| 13 | [Pradhan Mantri Kaushal Vikas Yojana - Short Term Training](https://www.myscheme.gov.in/schemes/pmkvy-stt) | Ministry Of Skill Development And Entrepreneurship | 2026-10-02 |
| 14 | [Pradhan Mantri Suraksha Bima Yojana](https://www.myscheme.gov.in/schemes/pmsby) | Ministry Of Finance | 2026-10-02 |
| 15 | [Stand-Up India](https://www.myscheme.gov.in/schemes/sui) | Ministry Of Finance | 2026-10-02 |

---

## 1. Atal Pension Yojana (APY)

- **Scheme id:** `atal_pension_yojana`
- **Ministry:** Ministry Of Finance
- **Source URL:** <https://www.myscheme.gov.in/schemes/apy>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** (i) Guaranteed minimum pension amount: Each subscriber under APY shall receive a  guaranteed minimum pension of Rs. 1000/- per month or Rs. 2000/- per month or Rs. 3000/- per month or Rs. 4000/- per month or Rs. 5000/- per month, after the age of 60 years until death.
- [ ] **Eligibility:** The minimum age of joining APY is 18 years and maximum is 40 years.
- [ ] **Minimum age: 18**
      - official wording: *"The minimum age of joining APY is 18 years and maximum is 40 years."*
- [ ] **Maximum age: 40**
      - official wording: *"The minimum age of joining APY is 18 years and maximum is 40 years."*

<details><summary>Conditions that could not be encoded as rules (3) — these are shown to users as 'still to verify'</summary>

- Must be a savings bank account holder.
- The official overview describes APY as being for a savings account holder 'who is not an income tax-payee'. This is a tax-status condition, not a stated income ceiling, so no numeric income limit is encoded.
- Contributions must be made by auto-debit from the savings bank account until the age of 60.

</details>

---

## 2. Kisan Credit Card (KCC)

- **Scheme id:** `kisan_credit_card`
- **Ministry:** Ministry Of Agriculture and Farmers Welfare
- **Source URL:** <https://www.myscheme.gov.in/schemes/kcc>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Fixation of credit limit/Loan amount
- [ ] **Eligibility:** Farmers - individual/joint borrowers who are owner cultivators;
- [ ] **Allowed occupations: farmer**
      - official wording: *"Farmers - individual/joint borrowers who are owner cultivators; tenant farmers, oral lessees & share croppers; Self Help Groups (SHGs) or Joint Liability Groups (JLGs) of farmers."*

<details><summary>Conditions that could not be encoded as rules (1) — these are shown to users as 'still to verify'</summary>

- Eligible as an individual/joint borrower who is an owner cultivator, a tenant farmer, oral lessee or share cropper, or as a member of an SHG or Joint Liability Group of farmers.

</details>

---

## 3. National Means-cum-Merit Scholarship Scheme (NMMSS)

- **Scheme id:** `nmmss`
- **Ministry:** Ministry of Education
- **Source URL:** <https://www.myscheme.gov.in/schemes/nmmss>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Under this scheme, 1,00,000 fresh scholarships @ ₹12,000/- per annum  (₹1,000 per month) are awarded to the meritorious students every year at Class IX level, which can be continued upto Class Xll.
- [ ] **Eligibility:** The applicant must be a student.
- [ ] **Maximum annual income: 350,000**
      - official wording: *"Students whose parental income from all sources is not more than Rs 3,50,000/- per annum are eligible to avail the scholarship."*
- [ ] **Allowed occupations: student**
      - official wording: *"The applicant must be a student."*

<details><summary>Conditions that could not be encoded as rules (4) — these are shown to users as 'still to verify'</summary>

- Must have a minimum of 55% marks or equivalent grade in the Class VII examination to appear in the selection test (relaxable by 5% for SC/ST students).
- Must be studying as a regular student in a Government, Government-aided or local body school.
- Must pass both the Mental Ability Test and the Scholastic Aptitude Test with at least 40% marks in aggregate (32% for SC/ST students).
- Reservation applies as per State/UT Government norms.

</details>

---

## 4. Pradhan Mantri Fasal Bima Yojna (PMFBY) (PMFBY)

- **Scheme id:** `pm_fasal_bima`
- **Ministry:** Ministry Of Agriculture and Farmers Welfare
- **Source URL:** <https://www.myscheme.gov.in/schemes/pmfby>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Affordable Premiums:**** The maximum premium payable by the farmer will be 2% for the Kharif food and oilseed crops. For rabi food and oilseeds crop, it is 1.5% and for yearly commercial or horticultural crops it will be 5%. The remaining premium is subsidized by the government.
- [ ] **Eligibility:** All farmers, including tenant farmers and sharecroppers growing notified crops in notified areas.
- [ ] **Allowed occupations: farmer**
      - official wording: *"All farmers, including tenant farmers and sharecroppers growing notified crops in notified areas."*

<details><summary>Conditions that could not be encoded as rules (5) — these are shown to users as 'still to verify'</summary>

- Must grow notified crops in notified areas.
- Must have an insurable interest in the insured crops.
- Must possess a valid and authenticated land ownership certificate or a valid land tenure agreement.
- Must apply within the prescribed time frame, usually within 2 weeks of the start of the sowing season.
- Must not have received compensation for the same crop loss from any other medium or source.

</details>

---

## 5. Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)

- **Scheme id:** `pm_kisan`
- **Ministry:** Ministry Of Agriculture and Farmers Welfare
- **Source URL:** <https://www.myscheme.gov.in/schemes/pm-kisan>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Financial benefit of Rs. 6000 per annum per family payable in three equal installments of Rs 2000 each, every four months.
- [ ] **Eligibility:** All landholding farmers' families, which have cultivable land holding in their names are eligible to get benefit under the scheme.
- [ ] **Allowed occupations: farmer**
      - official wording: *"All landholding farmers' families, which have cultivable land holding in their names are eligible to get benefit under the scheme."*

<details><summary>Conditions that could not be encoded as rules (2) — these are shown to users as 'still to verify'</summary>

- Must have cultivable land holding in their own name.
- Excluded: institutional land holders; income tax payers in the last assessment year; serving/retired government employees (except Group D / Class IV / Multi Tasking Staff); retired pensioners with monthly pension of Rs.10,000 or more; present and former holders of constitutional posts; registered practising professionals (doctors, engineers, lawyers, chartered accountants, architects).

</details>

---

## 6. Pradhan Mantri Matru Vandana Yojana (PMMVY)

- **Scheme id:** `pm_matru_vandana`
- **Ministry:** Ministry of Women and Child Development
- **Source URL:** <https://www.myscheme.gov.in/schemes/pmmvy>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** The cash incentives is provided in two instalments for the first child as per the schedule provided in
- [ ] **Eligibility:** The applicant should be of at least 19 years old and a pregnant women.
- [ ] **Minimum age: 19**
      - official wording: *"The applicant should be of at least 19 years old and a pregnant women."*
- [ ] **Gender restriction: female**
      - official wording: *"The applicant should be of at least 19 years old and a pregnant women."*

<details><summary>Conditions that could not be encoded as rules (4) — these are shown to users as 'still to verify'</summary>

- Must be pregnant, employed, and experiencing wage-loss due to the pregnancy.
- Applicable only for the first live birth (with a separate provision for a second girl child in the case of twins/triplets/quadruplets).
- Must apply within 270 days from the child's birth.
- Must fall in one of the listed socially/economically disadvantaged categories. 'Net family income less than Rs 8 Lakh per annum' is only ONE of those alternative routes, so it is NOT encoded as a universal income ceiling.

</details>

---

## 7. Pradhan Mantri Mudra Yojana (PMMY)

- **Scheme id:** `pm_mudra`
- **Ministry:** Ministry Of Finance
- **Source URL:** <https://www.myscheme.gov.in/schemes/pmmy>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** The scheme has been classified under four categories as 'SHISHU', 'KISHORE' , 'TARUN' and 'TARUN PLUS' to signify the stage of growth / development and funding needs of the beneficiary micro unit/ entrepreneur.
- [ ] **Eligibility:** Note 01:The applicant should not be a defaulter to any bank or financial institution and should have a satisfactory credit track record.
- [ ] **No age / income / state / occupation / gender / category rule is encoded for this scheme** — confirm the official page really states no such restriction.

<details><summary>Conditions that could not be encoded as rules (3) — these are shown to users as 'still to verify'</summary>

- Eligible borrowers: individuals, proprietary concerns, partnership firms, private limited companies, public companies and any other legal forms.
- The applicant should not be a defaulter to any bank or financial institution and should have a satisfactory credit track record.
- Individual borrowers may be required to possess the necessary skills, experience or knowledge to undertake the proposed activity.

</details>

---

## 8. PM Street Vendor’s AtmaNirbhar Nidhi (PM SVANidhi) (PM SVANIDHI)

- **Scheme id:** `pm_svanidhi`
- **Ministry:** Ministry Of Housing & Urban Affairs
- **Source URL:** <https://www.myscheme.gov.in/schemes/pm-svanidhi>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Financial assistance of up to ₹10,000 is provided to street vendors to help them restart and expand their businesses.
- [ ] **Eligibility:** Street vendors in possession of Certificate of Vending / Identity Card issued by Urban Local Bodies (ULBs).
- [ ] **Allowed occupations: street_vendor**
      - official wording: *"Street vendors in possession of Certificate of Vending / Identity Card issued by Urban Local Bodies (ULBs)."*

<details><summary>Conditions that could not be encoded as rules (1) — these are shown to users as 'still to verify'</summary>

- Must hold a Certificate of Vending / Identity Card from the Urban Local Body, or have been identified in the ULB survey, or hold a Letter of Recommendation from the ULB / Town Vending Committee.

</details>

---

## 9. Pradhan Mantri Ujjwala Yojana (PMUY)

- **Scheme id:** `pm_ujjwala`
- **Ministry:** Ministry Of Petroleum and Natural Gas
- **Source URL:** <https://www.myscheme.gov.in/schemes/pmuy>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Rs. 1600 for a connection 14.2kg cylinder or Rs. 1150 for a 5 kg cylinder.
- [ ] **Eligibility:** Eligible as per SECC 2011 list
- [ ] **Minimum age: 18**
      - official wording: *"Derived from 'An adult woman...' in the official eligibility text. The source says 'adult' rather than a number; 18 is used as the statutory age of adulthood in India."*
- [ ] **Gender restriction: female**
      - official wording: *"An adult woman belonging to a poor household and not having an LPG connection in her household will be eligible under UJJWALA 2.0."*

<details><summary>Conditions that could not be encoded as rules (2) — these are shown to users as 'still to verify'</summary>

- Must belong to a poor household and must not already have an LPG connection in the household.
- Must qualify through ONE of: the SECC 2011 list; SC/ST household, PMAY / Antyodaya Anna Yojana beneficiary, forest dweller, Most Backward Class, Tea or Ex-Tea Garden Tribe, or resident of a river island; or a 14-point declaration as a poor household. Because these are alternatives, no single category is encoded as a hard filter.

</details>

---

## 10. PM Vishwakarma (PMV)

- **Scheme id:** `pm_vishwakarma`
- **Ministry:** Ministry Of Micro, Small and Medium Enterprises
- **Source URL:** <https://www.myscheme.gov.in/schemes/pmv>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Skill Verification followed by 5-7 days (40 hours) of Basic Training
- [ ] **Eligibility:** The applicant should be an artisan or craftsperson working with hands and tools.
- [ ] **Minimum age: 18**
      - official wording: *"On the date of registration for the scheme, the minimum age of the applicant should be 18 years."*
- [ ] **Allowed occupations: artisan, craftsperson**
      - official wording: *"The applicant should be an artisan or craftsperson working with hands and tools."*

<details><summary>Conditions that could not be encoded as rules (4) — these are shown to users as 'still to verify'</summary>

- Must be engaged in the unorganized sector on a self-employment basis.
- Must be engaged in one of the 18 family-based traditional trades listed in the scheme.
- Must not have availed loans under similar central or state credit-based self-employment schemes (e.g. PMEGP, PM SVANidhi, Mudra) in the past 5 years.
- Benefits are restricted to one member of the family.

</details>

---

## 11. Pradhan Mantri Awas Yojana - Urban (PMAY - U)

- **Scheme id:** `pmay_urban`
- **Ministry:** Ministry Of Housing & Urban Affairs
- **Source URL:** <https://www.myscheme.gov.in/schemes/pmay-u>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Slum rehabilitation of eligible Slum Dwellers with participation of private developers using land as a resource.
- [ ] **Eligibility:** The family identifies as one of the following -
- [ ] **Maximum annual income: 1,800,000**
      - official wording: *"Middle Income Group-2 (MIG-2): households with annual income between Rs 12,00,001 and Rs 18,00,000. (Rs 18,00,000 is the highest household income covered by any of the scheme's four income categories: EWS up to Rs 3,00,000; LIG Rs 3,00,001-6,00,000; MIG-1 Rs 6,00,001-12,00,000; MIG-2 Rs 12,00,001-18,00,000.)"*

<details><summary>Conditions that could not be encoded as rules (5) — these are shown to users as 'still to verify'</summary>

- The applicant or their family members must not own a pucca house anywhere in the country.
- The family must comprise husband/wife and unmarried children.
- The town/city where the family resides must be covered under the scheme.
- The family must not have previously availed benefits of any housing-related scheme of the Government of India.
- Income band determines which vertical of the scheme applies (EWS, LIG, MIG-1, MIG-2); this engine only checks the overall Rs 18,00,000 ceiling.

</details>

---

## 12. Pradhan Mantri Jeevan Jyoti Bima Yojana (PMJJBY)

- **Scheme id:** `pmjjby`
- **Ministry:** Ministry Of Finance
- **Source URL:** <https://www.myscheme.gov.in/schemes/pmjjby>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** PMJJBY offers one- year term life cover of ₹ 2.00 Lakh to all the subscribers in the age group of 18-50 years.1. It covers death due to any reason.
- [ ] **Eligibility:** The age of the applicant must be between 18 to 50 Years.
- [ ] **Minimum age: 18**
      - official wording: *"The age of the applicant must be between 18 to 50 Years."*
- [ ] **Maximum age: 50**
      - official wording: *"The age of the applicant must be between 18 to 50 Years."*

<details><summary>Conditions that could not be encoded as rules (1) — these are shown to users as 'still to verify'</summary>

- Must hold an individual bank / post office account.

</details>

---

## 13. Pradhan Mantri Kaushal Vikas Yojana - Short Term Training (PMKVY -STT)

- **Scheme id:** `pmkvy_stt`
- **Ministry:** Ministry Of Skill Development And Entrepreneurship
- **Source URL:** <https://www.myscheme.gov.in/schemes/pmkvy-stt>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Online Information / Counselling Platform
- [ ] **Eligibility:** Is aged between 15-45 years
- [ ] **Minimum age: 15**
      - official wording: *"Is aged between 15-45 years"*
- [ ] **Maximum age: 45**
      - official wording: *"Is aged between 15-45 years"*

<details><summary>Conditions that could not be encoded as rules (3) — these are shown to users as 'still to verify'</summary>

- Must be of Indian nationality.
- Must possess an Aadhaar card and an Aadhaar-linked bank account.
- Must fulfil other criteria for the respective job role as defined by the awarding body.

</details>

---

## 14. Pradhan Mantri Suraksha Bima Yojana (PMSBY)

- **Scheme id:** `pmsby`
- **Ministry:** Ministry Of Finance
- **Source URL:** <https://www.myscheme.gov.in/schemes/pmsby>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** On Death- the Nominee shall get Rs. 2 Lakhs.
- [ ] **Eligibility:** Individual bank account holders of participating banks aged between 18 years (completed) and 70 years (age nearer birthday) who give their consent to join / enable auto-debit, will be enrolled into the scheme.
- [ ] **Minimum age: 18**
      - official wording: *"Individual bank account holders of participating banks aged between 18 years (completed) and 70 years (age nearer birthday)."*
- [ ] **Maximum age: 70**
      - official wording: *"Individual bank account holders of participating banks aged between 18 years (completed) and 70 years (age nearer birthday)."*

<details><summary>Conditions that could not be encoded as rules (1) — these are shown to users as 'still to verify'</summary>

- Must be an individual bank account holder of a participating bank and give consent to join / enable auto-debit.

</details>

---

## 15. Stand-Up India (SUPI)

- **Scheme id:** `stand_up_india`
- **Ministry:** Ministry Of Finance
- **Source URL:** <https://www.myscheme.gov.in/schemes/sui>
- **Last verified:** 2026-10-02

**Check these facts:**

- [ ] **Benefit:** Facilitation of composite loan (inclusive of term loan and working capital) between ₹10 Lakhs and ₹100 Lakhs. Rupay debit card to be issued for convenience of the borrower.
- [ ] **Eligibility:** Finance is  provided for Greenfield Enterprises.
- [ ] **Minimum age: 18**
      - official wording: *"The age of the applicant must be at least 18 years."*

<details><summary>Conditions that could not be encoded as rules (3) — these are shown to users as 'still to verify'</summary>

- CONDITIONAL RULE: 'If the applicant is a male, he must be from SC / ST category.' Women applicants are eligible regardless of category. A flat gender/category filter cannot express this, so neither field is set and the engine reports this condition for human checking.
- Finance is provided for Greenfield Enterprises (first-time venture).
- The applicant must not be in default to any bank / financial institution.

</details>

---
