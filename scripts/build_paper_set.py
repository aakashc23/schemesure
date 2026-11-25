"""
Build eval/paper_set.jsonl — the translation-paired benchmark for the paper.

WHY A SECOND SET (and why the golden set is left untouched)
eval/golden_set.jsonl (50 questions) is what every existing result was measured
on, so it stays frozen. This set exists to fix the two weaknesses the evaluation
review found in it:

  1. Language was confounded with difficulty: the Hindi and Hinglish questions
     were *different* questions from the English ones. Here every question is
     written in all three languages and the three versions share a `pair_id`,
     so a per-language comparison holds the content fixed.
  2. n was too small per failure mode (1-5). Here each unanswerable failure
     mode has 10-15 distinct questions, x3 languages.

WHAT IS CHECKED (the build fails otherwise)
  - answerable: every key_fact appears in the expected scheme's source file
    (same check as build_golden_set.py).
  - not_indexed: the named scheme is not one of the 15 indexed schemes.
  - trick_unstated_fact: the question targets an indexed scheme, and each
    `absent` regex — a pattern that would have to match if the source DID state
    the answer — matches nothing in that scheme's body text. This makes "the
    source does not say this" a checked property instead of an assumption.
  - every pair has exactly one question per language.

WHAT IS NOT CHECKED, and must be said in the paper
  - Translations were written by the authors, not by professional translators,
    and have not yet been reviewed by a second native speaker.
  - "fake_scheme" names were checked against the 15 indexed schemes and the
    authors' knowledge, not against an exhaustive list of every state scheme.
  - The `absent` patterns are necessary, not sufficient: they rule out the
    obvious phrasings of an answer, not every possible one.

Run:  python scripts/build_paper_set.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_golden_set import fact_present, normalize  # noqa: E402

SCHEMES_DIR = ROOT / "data" / "schemes"
OUT_PATH = ROOT / "eval" / "paper_set.jsonl"
LANGUAGES = ("en", "hi", "hinglish")

# ==========================================================================
# ANSWERABLE (35 concepts x 3 languages = 105)
# (pair_id, scheme, key_facts, category, {en, hi, hinglish})
# The 35 concepts are the golden set's answerable questions; the version in the
# golden set's original language is reused verbatim.
# ==========================================================================

ANSWERABLE: list[tuple[str, str, list[str], str, dict[str, str]]] = [
    # ---- PM-KISAN ----
    ("pa01", "pm_kisan", ["6000"], "benefit_amount", {
        "en": "How much money does PM-KISAN give per year?",
        "hi": "पीएम किसान योजना में किसानों को साल में कितने रुपये मिलते हैं?",
        "hinglish": "PM Kisan mein saal ka kitna paisa milta hai?",
    }),
    ("pa02", "pm_kisan", ["three equal installments", "2000"], "benefit_structure", {
        "en": "In how many instalments is the PM-KISAN amount paid?",
        "hi": "पीएम किसान की राशि कितनी किस्तों में दी जाती है?",
        "hinglish": "PM Kisan ka paisa kitne installment mein milta hai?",
    }),
    ("pa03", "pm_kisan", ["paid Income Tax in last assessment year"], "exclusion", {
        "en": "Are income tax payers eligible for PM-KISAN?",
        "hi": "क्या आयकर देने वाले लोग पीएम किसान के पात्र हैं?",
        "hinglish": "Kya income tax bharne wale PM Kisan ke liye eligible hain?",
    }),
    # ---- PMJJBY ----
    ("pa04", "pmjjby", ["2.00 Lakh"], "benefit_amount", {
        "en": "What is the life cover amount under Pradhan Mantri Jeevan Jyoti Bima Yojana?",
        "hi": "प्रधानमंत्री जीवन ज्योति बीमा योजना में जीवन बीमा कवर कितना है?",
        "hinglish": "PM Jeevan Jyoti Bima Yojana mein life cover kitna milta hai?",
    }),
    ("pa05", "pmjjby", ["436"], "cost", {
        "en": "What is the annual premium for PMJJBY?",
        "hi": "पीएमजेजेबीवाई का सालाना प्रीमियम कितना है?",
        "hinglish": "PMJJBY ka saal ka premium kitna hai?",
    }),
    ("pa06", "pmjjby", ["18 to 50 Years"], "eligibility_age", {
        "en": "What is the age limit for PM Jeevan Jyoti Bima Yojana?",
        "hi": "पीएम जीवन ज्योति बीमा योजना के लिए आयु सीमा क्या है?",
        "hinglish": "PM Jeevan Jyoti Bima Yojana ke liye age limit kya hai?",
    }),
    # ---- PMSBY ----
    ("pa07", "pmsby", ["2 Lakhs"], "benefit_amount", {
        "en": "How much does the nominee get on death under PM Suraksha Bima Yojana?",
        "hi": "पीएम सुरक्षा बीमा योजना में मृत्यु होने पर नॉमिनी को कितना मिलता है?",
        "hinglish": "PM Suraksha Bima Yojana mein death hone par nominee ko kitna milta hai?",
    }),
    ("pa08", "pmsby", ["70 years"], "eligibility_age", {
        "en": "What is the maximum age to join Pradhan Mantri Suraksha Bima Yojana?",
        "hi": "प्रधानमंत्री सुरक्षा बीमा योजना में शामिल होने की अधिकतम आयु क्या है?",
        "hinglish": "PM Suraksha Bima Yojana join karne ki maximum age kya hai?",
    }),
    ("pa09", "pmsby", ["1 Lakh"], "benefit_amount", {
        "en": "How much does PMSBY pay for the loss of sight of one eye?",
        "hi": "पीएमएसबीवाई में एक आंख की रोशनी जाने पर कितना पैसा मिलता है?",
        "hinglish": "PMSBY mein ek aankh ki roshni jaane par kitna milta hai?",
    }),
    # ---- Atal Pension Yojana ----
    ("pa10", "atal_pension_yojana", ["18 years and maximum is 40 years"], "eligibility_age", {
        "en": "What is the minimum and maximum age to join Atal Pension Yojana?",
        "hi": "अटल पेंशन योजना में शामिल होने की न्यूनतम और अधिकतम आयु क्या है?",
        "hinglish": "Atal Pension Yojana join karne ki minimum aur maximum age kya hai?",
    }),
    ("pa11", "atal_pension_yojana", ["60 years"], "benefit_timing", {
        "en": "From what age does the pension start under Atal Pension Yojana?",
        "hi": "अटल पेंशन योजना में पेंशन कब से शुरू होती है?",
        "hinglish": "Atal Pension Yojana mein pension kis umar se shuru hoti hai?",
    }),
    # ---- PM Ujjwala ----
    ("pa12", "pm_ujjwala", ["1600"], "benefit_amount", {
        "en": "How much deposit support is given for a 14.2 kg cylinder under PM Ujjwala?",
        "hi": "पीएम उज्ज्वला योजना में 14.2 किलो सिलेंडर के कनेक्शन के लिए कितनी राशि दी जाती है?",
        "hinglish": "Ujjwala yojana mein 14.2 kg cylinder connection ke liye kitna paisa milta hai?",
    }),
    ("pa13", "pm_ujjwala", ["Aadhaar"], "documents", {
        "en": "What documents are required for PM Ujjwala?",
        "hi": "पीएम उज्ज्वला योजना के लिए कौन से दस्तावेज़ चाहिए?",
        "hinglish": "Ujjwala yojana ke liye kaun se documents chahiye?",
    }),
    ("pa14", "pm_ujjwala", ["adult woman"], "eligibility", {
        "en": "Who is eligible for the Ujjwala scheme?",
        "hi": "उज्ज्वला योजना के लिए कौन पात्र है?",
        "hinglish": "Ujjwala yojana ke liye kaun eligible hai?",
    }),
    # ---- NMMSS ----
    ("pa15", "nmmss", ["3,50,000"], "eligibility_income", {
        "en": "What is the parental income limit for the National Means-cum-Merit Scholarship?",
        "hi": "राष्ट्रीय साधन-सह-मेधा छात्रवृत्ति के लिए माता-पिता की आय सीमा क्या है?",
        "hinglish": "NMMSS scholarship ke liye parents ki income limit kya hai?",
    }),
    ("pa16", "nmmss", ["12,000"], "benefit_amount", {
        "en": "How much scholarship is given per year under NMMSS?",
        "hi": "एनएमएमएसएस में हर साल कितनी छात्रवृत्ति मिलती है?",
        "hinglish": "NMMSS mein saal ki kitni scholarship milti hai?",
    }),
    ("pa17", "nmmss", ["55 %"], "eligibility_marks", {
        "en": "What minimum marks are needed in Class VII to apply for NMMSS?",
        "hi": "एनएमएमएसएस के लिए कक्षा 7 में न्यूनतम कितने अंक चाहिए?",
        "hinglish": "NMMSS ke liye class 7 mein kam se kam kitne marks chahiye?",
    }),
    # ---- PM SVANidhi ----
    ("pa18", "pm_svanidhi", ["10,000"], "benefit_amount", {
        "en": "How much loan can a street vendor get under PM SVANidhi?",
        "hi": "पीएम स्वनिधि में रेहड़ी-पटरी वाले को कितना लोन मिल सकता है?",
        "hinglish": "PM SVANidhi mein street vendor ko kitna loan mil sakta hai?",
    }),
    ("pa19", "pm_svanidhi", ["7%"], "cost", {
        "en": "What is the interest rate on the PM SVANidhi loan?",
        "hi": "पीएम स्वनिधि लोन पर ब्याज दर क्या है?",
        "hinglish": "PM SVANidhi loan ka interest rate kya hai?",
    }),
    ("pa20", "pm_svanidhi", ["Street vendors"], "eligibility", {
        "en": "Who can apply for PM SVANidhi?",
        "hi": "पीएम स्वनिधि के लिए कौन आवेदन कर सकता है?",
        "hinglish": "PM SVANidhi ke liye kaun apply kar sakta hai?",
    }),
    # ---- PM Mudra ----
    ("pa21", "pm_mudra", ["Shishu", "Kishore", "Tarun"], "benefit_structure", {
        "en": "What are the loan categories under Pradhan Mantri Mudra Yojana?",
        "hi": "प्रधानमंत्री मुद्रा योजना में लोन की कौन-कौन सी श्रेणियां हैं?",
        "hinglish": "Mudra Yojana mein loan ki kaun kaun si categories hain?",
    }),
    ("pa22", "pm_mudra", ["10 lakhs"], "benefit_amount", {
        "en": "What is the maximum loan under the Tarun category of Mudra Yojana?",
        "hi": "मुद्रा योजना की तरुण श्रेणी में अधिकतम कितना लोन मिलता है?",
        "hinglish": "Mudra Yojana ki Tarun category mein maximum kitna loan milta hai?",
    }),
    ("pa23", "pm_mudra", ["50,000"], "benefit_amount", {
        "en": "How much loan is given under the Shishu category of Mudra Yojana?",
        "hi": "मुद्रा योजना की शिशु श्रेणी में कितना ऋण मिलता है?",
        "hinglish": "Mudra Yojana ki Shishu category mein kitna loan milta hai?",
    }),
    # ---- Stand-Up India ----
    ("pa24", "stand_up_india", ["10 Lakhs", "100 Lakhs"], "benefit_amount", {
        "en": "What is the loan range under Stand-Up India?",
        "hi": "स्टैंड-अप इंडिया में लोन की सीमा कितनी है?",
        "hinglish": "Stand-Up India mein loan kitne se kitne tak milta hai?",
    }),
    ("pa25", "stand_up_india", ["SC / ST category"], "eligibility", {
        "en": "Who is eligible for a Stand-Up India loan if the applicant is male?",
        "hi": "अगर आवेदक पुरुष है तो स्टैंड-अप इंडिया लोन के लिए कौन पात्र है?",
        "hinglish": "Agar applicant male hai to Stand-Up India loan ke liye kaun eligible hai?",
    }),
    # ---- PM Vishwakarma ----
    ("pa26", "pm_vishwakarma", ["15,000"], "benefit_amount", {
        "en": "What is the toolkit incentive under PM Vishwakarma?",
        "hi": "पीएम विश्वकर्मा योजना में टूलकिट प्रोत्साहन कितना है?",
        "hinglish": "PM Vishwakarma mein toolkit ke liye kitna paisa milta hai?",
    }),
    ("pa27", "pm_vishwakarma", ["500 per day"], "benefit_amount", {
        "en": "What training stipend is given under PM Vishwakarma?",
        "hi": "पीएम विश्वकर्मा में प्रशिक्षण के दौरान कितना स्टाइपेंड मिलता है?",
        "hinglish": "PM Vishwakarma mein training ke time kitna stipend milta hai?",
    }),
    # ---- PMAY Urban ----
    ("pa28", "pmay_urban", ["3,00,000"], "eligibility_income", {
        "en": "What is the EWS annual household income limit under PMAY Urban?",
        "hi": "पीएमएवाई शहरी में ईडब्ल्यूएस परिवार की सालाना आय सीमा क्या है?",
        "hinglish": "PMAY Urban mein EWS ki saalana income limit kya hai?",
    }),
    ("pa29", "pmay_urban", ["18,00,000"], "eligibility_income", {
        "en": "What is the income limit for MIG-2 under PMAY Urban?",
        "hi": "पीएमएवाई शहरी में एमआईजी-2 की आय सीमा क्या है?",
        "hinglish": "PMAY Urban mein MIG 2 ki income limit kya hai?",
    }),
    # ---- PMMVY ----
    ("pa30", "pm_matru_vandana", ["3,000"], "benefit_amount", {
        "en": "How much is the first instalment under Pradhan Mantri Matru Vandana Yojana?",
        "hi": "प्रधानमंत्री मातृ वंदना योजना की पहली किस्त कितनी है?",
        "hinglish": "PM Matru Vandana Yojana ki pehli kist kitni hai?",
    }),
    ("pa31", "pm_matru_vandana", ["19 years"], "eligibility_age", {
        "en": "What is the minimum age to apply for PM Matru Vandana Yojana?",
        "hi": "पीएम मातृ वंदना योजना के लिए न्यूनतम आयु कितनी है?",
        "hinglish": "PM Matru Vandana Yojana ke liye kam se kam umar kitni honi chahiye?",
    }),
    # ---- PMKVY ----
    ("pa32", "pmkvy_stt", ["15-45 years"], "eligibility_age", {
        "en": "What is the age range for Pradhan Mantri Kaushal Vikas Yojana short term training?",
        "hi": "प्रधानमंत्री कौशल विकास योजना के अल्पकालिक प्रशिक्षण के लिए आयु सीमा क्या है?",
        "hinglish": "PM Kaushal Vikas Yojana short term training ke liye age limit kya hai?",
    }),
    # ---- PMFBY ----
    ("pa33", "pm_fasal_bima", ["2%"], "cost", {
        "en": "What premium does a farmer pay for Kharif crops under PM Fasal Bima Yojana?",
        "hi": "पीएम फसल बीमा योजना में खरीफ फसल के लिए किसान को कितना प्रीमियम देना होता है?",
        "hinglish": "PM Fasal Bima Yojana mein Kharif fasal ka premium kitna hai?",
    }),
    ("pa34", "pm_fasal_bima", ["1.5%"], "cost", {
        "en": "What premium does a farmer pay for Rabi crops under PM Fasal Bima Yojana?",
        "hi": "पीएम फसल बीमा योजना में रबी फसल के लिए किसान को कितना प्रीमियम देना होता है?",
        "hinglish": "PM Fasal Bima Yojana mein Rabi fasal ka premium kitna hai?",
    }),
    # ---- Kisan Credit Card ----
    ("pa35", "kisan_credit_card", ["owner cultivators"], "eligibility", {
        "en": "Who is eligible for a Kisan Credit Card?",
        "hi": "किसान क्रेडिट कार्ड के लिए कौन पात्र है?",
        "hinglish": "Kisan Credit Card ke liye kaun eligible hai?",
    }),
]

# ==========================================================================
# UNANSWERABLE
# (pair_id, category, {en, hi, hinglish}, extra)
#   off_topic            nothing to do with welfare schemes
#   fake_scheme          a scheme that does not exist, with a plausible name
#   not_indexed          a REAL central scheme that is not in the 15 indexed;
#                        extra = {"real_scheme": ...}. Several deliberately share
#                        words with an indexed scheme (PMAY-Gramin vs PMAY-Urban,
#                        PM Kisan Maandhan vs PM-KISAN): the near-miss case.
#   trick_unstated_fact  an indexed scheme, asking for a fact its source does not
#                        state; extra = {"scheme": ..., "absent": [regex, ...]}
# ==========================================================================

OFF_TOPIC: list[tuple[str, dict[str, str]]] = [
    ("po01", {"en": "What is the capital of France?",
              "hi": "फ्रांस की राजधानी क्या है?",
              "hinglish": "France ki capital kya hai?"}),
    ("po02", {"en": "Give me a good recipe for pizza dough.",
              "hi": "पिज़्ज़ा का आटा बनाने की अच्छी विधि बताइए।",
              "hinglish": "Pizza dough banane ki achhi recipe batao."}),
    ("po03", {"en": "How do I reset my Gmail password?",
              "hi": "मैं अपना जीमेल पासवर्ड कैसे रीसेट करूं?",
              "hinglish": "Gmail ka password reset kaise karu?"}),
    ("po04", {"en": "Who won the ICC Cricket World Cup in 2011?",
              "hi": "2011 का आईसीसी क्रिकेट विश्व कप किसने जीता?",
              "hinglish": "2011 ka cricket world cup kisne jeeta tha?"}),
    ("po05", {"en": "What is the national sport of India?",
              "hi": "भारत का राष्ट्रीय खेल कौन सा है?",
              "hinglish": "India ka national sport kaunsa hai?"}),
    ("po06", {"en": "How many planets are there in the solar system?",
              "hi": "सौरमंडल में कितने ग्रह हैं?",
              "hinglish": "Solar system mein kitne planet hain?"}),
    ("po07", {"en": "What is a good home remedy for a cold?",
              "hi": "सर्दी-जुकाम का अच्छा घरेलू उपाय क्या है?",
              "hinglish": "Sardi zukam ka koi achha gharelu ilaj batao."}),
    ("po08", {"en": "Write a short poem about the monsoon.",
              "hi": "मानसून पर एक छोटी कविता लिखिए।",
              "hinglish": "Monsoon par ek chhoti si kavita likho."}),
    ("po09", {"en": "How do I learn Python programming quickly?",
              "hi": "मैं पायथन प्रोग्रामिंग जल्दी कैसे सीखूं?",
              "hinglish": "Python programming jaldi kaise seekhu?"}),
    ("po10", {"en": "What is the boiling point of water?",
              "hi": "पानी का क्वथनांक कितना होता है?",
              "hinglish": "Paani kitne degree par ubalta hai?"}),
    ("po11", {"en": "Which movie won the Oscar for best picture in 2020?",
              "hi": "2020 में सर्वश्रेष्ठ फिल्म का ऑस्कर किस फिल्म को मिला?",
              "hinglish": "2020 mein best picture ka Oscar kis movie ko mila?"}),
    ("po12", {"en": "How do I convert Celsius to Fahrenheit?",
              "hi": "सेल्सियस को फ़ारेनहाइट में कैसे बदलें?",
              "hinglish": "Celsius ko Fahrenheit mein kaise convert karte hain?"}),
    ("po13", {"en": "What is the tallest mountain in the world?",
              "hi": "दुनिया का सबसे ऊंचा पर्वत कौन सा है?",
              "hinglish": "Duniya ka sabse uncha pahad kaunsa hai?"}),
    ("po14", {"en": "Suggest a good name for my pet dog.",
              "hi": "मेरे पालतू कुत्ते के लिए एक अच्छा नाम सुझाइए।",
              "hinglish": "Mere dog ke liye koi achha naam batao."}),
    ("po15", {"en": "How long should I boil an egg?",
              "hi": "अंडे को कितनी देर उबालना चाहिए?",
              "hinglish": "Anda kitni der ubalna chahiye?"}),
]

FAKE_SCHEME: list[tuple[str, dict[str, str]]] = [
    ("pf01", {"en": "Tell me about the Pradhan Mantri Free Laptop Yojana 2026.",
              "hi": "प्रधानमंत्री फ्री लैपटॉप योजना 2026 के बारे में बताइए।",
              "hinglish": "PM Free Laptop Yojana 2026 ke baare mein batao."}),
    ("pf02", {"en": "How much money does the PM Berojgari Bhatta Yojana give every month?",
              "hi": "पीएम बेरोजगारी भत्ता योजना में हर महीने कितने पैसे मिलते हैं?",
              "hinglish": "PM Berojgari Bhatta Yojana mein har mahine kitna paisa milta hai?"}),
    ("pf03", {"en": "How do I apply for the Pradhan Mantri Free Scooty Yojana?",
              "hi": "प्रधानमंत्री मुफ्त स्कूटी योजना के लिए आवेदन कैसे करें?",
              "hinglish": "PM Free Scooty Yojana ke liye apply kaise karein?"}),
    ("pf04", {"en": "How much money is given under the PM Shadi Shagun Yojana?",
              "hi": "पीएम शादी शगुन योजना में कितना पैसा मिलता है?",
              "hinglish": "PM Shadi Shagun Yojana mein kitna paisa milta hai?"}),
    ("pf05", {"en": "What is the eligibility for the National Digital Pension Scheme 2027?",
              "hi": "राष्ट्रीय डिजिटल पेंशन योजना 2027 की पात्रता क्या है?",
              "hinglish": "National Digital Pension Scheme 2027 ki eligibility kya hai?"}),
    ("pf06", {"en": "What is the monthly benefit under the PM Kisan Samriddhi Bonus Yojana?",
              "hi": "पीएम किसान समृद्धि बोनस योजना में हर महीने कितना लाभ मिलता है?",
              "hinglish": "PM Kisan Samriddhi Bonus Yojana mein monthly kitna milta hai?"}),
    ("pf07", {"en": "How much free mobile data does the Pradhan Mantri Free Internet Yojana give?",
              "hi": "प्रधानमंत्री फ्री इंटरनेट योजना में कितना मुफ्त मोबाइल डेटा मिलता है?",
              "hinglish": "PM Free Internet Yojana mein kitna free data milta hai?"}),
    ("pf08", {"en": "What is the loan amount under the Mudra Mahila Super Loan Yojana?",
              "hi": "मुद्रा महिला सुपर लोन योजना में कितना लोन मिलता है?",
              "hinglish": "Mudra Mahila Super Loan Yojana mein kitna loan milta hai?"}),
    ("pf09", {"en": "Who is eligible for the Atal Yuva Pension Scheme for students?",
              "hi": "छात्रों के लिए अटल युवा पेंशन योजना के लिए कौन पात्र है?",
              "hinglish": "Students ke liye Atal Yuva Pension Scheme mein kaun eligible hai?"}),
    ("pf10", {"en": "How much is the cash prize under the PM Gaon Ratna Puraskar Yojana?",
              "hi": "पीएम गांव रत्न पुरस्कार योजना में कितना नकद इनाम मिलता है?",
              "hinglish": "PM Gaon Ratna Puraskar Yojana mein kitna cash prize milta hai?"}),
    ("pf11", {"en": "What is the subsidy under the Pradhan Mantri Free Solar Cycle Yojana?",
              "hi": "प्रधानमंत्री फ्री सोलर साइकिल योजना में कितनी सब्सिडी मिलती है?",
              "hinglish": "PM Free Solar Cycle Yojana mein kitni subsidy milti hai?"}),
    ("pf12", {"en": "What is the life cover under the PM Jeevan Suraksha Gold Policy?",
              "hi": "पीएम जीवन सुरक्षा गोल्ड पॉलिसी में जीवन बीमा कवर कितना है?",
              "hinglish": "PM Jeevan Suraksha Gold Policy mein life cover kitna hai?"}),
    ("pf13", {"en": "How do I register for the National Street Vendor Pension Yojana?",
              "hi": "राष्ट्रीय स्ट्रीट वेंडर पेंशन योजना में पंजीकरण कैसे करें?",
              "hinglish": "National Street Vendor Pension Yojana mein registration kaise karein?"}),
    ("pf14", {"en": "What documents are needed for the PM Har Ghar Fridge Yojana?",
              "hi": "पीएम हर घर फ्रिज योजना के लिए कौन से दस्तावेज़ चाहिए?",
              "hinglish": "PM Har Ghar Fridge Yojana ke liye kaun se documents chahiye?"}),
    ("pf15", {"en": "How much scholarship does the Pradhan Mantri Topper Bonus Scheme give?",
              "hi": "प्रधानमंत्री टॉपर बोनस योजना में कितनी छात्रवृत्ति मिलती है?",
              "hinglish": "PM Topper Bonus Scheme mein kitni scholarship milti hai?"}),
]

NOT_INDEXED: list[tuple[str, str, dict[str, str]]] = [
    ("pn01", "Sukanya Samriddhi Yojana", {
        "en": "What is the current interest rate on Sukanya Samriddhi Yojana?",
        "hi": "सुकन्या समृद्धि योजना पर वर्तमान ब्याज दर क्या है?",
        "hinglish": "Sukanya Samriddhi Yojana ka abhi interest rate kya hai?"}),
    ("pn02", "Ayushman Bharat PM-JAY", {
        "en": "How do I get an Ayushman Bharat PM-JAY health card?",
        "hi": "आयुष्मान भारत पीएम-जय हेल्थ कार्ड कैसे बनवाएं?",
        "hinglish": "Ayushman Bharat card kaise banwayein?"}),
    ("pn03", "Ayushman Bharat PM-JAY", {
        "en": "Up to how many lakh rupees of treatment is free under Ayushman Bharat?",
        "hi": "आयुष्मान भारत में कितने लाख तक का इलाज मुफ्त है?",
        "hinglish": "Ayushman Bharat mein kitne lakh tak ka ilaj free hai?"}),
    ("pn04", "Atal Innovation Mission", {
        "en": "How much funding does a startup get under the Atal Innovation Mission?",
        "hi": "अटल इनोवेशन मिशन के तहत स्टार्टअप को कितनी फंडिंग मिलती है?",
        "hinglish": "Atal Innovation Mission ke under startup ko kitna funding milta hai?"}),
    ("pn05", "PMAY-Gramin", {
        "en": "How much assistance is given to build a house under PMAY-Gramin?",
        "hi": "पीएमएवाई-ग्रामीण में घर बनाने के लिए कितनी सहायता मिलती है?",
        "hinglish": "PMAY Gramin mein ghar banane ke liye kitna paisa milta hai?"}),
    ("pn06", "PM Kisan Maandhan Yojana", {
        "en": "What monthly pension does PM Kisan Maandhan Yojana give?",
        "hi": "पीएम किसान मानधन योजना में हर महीने कितनी पेंशन मिलती है?",
        "hinglish": "PM Kisan Maandhan Yojana mein monthly kitni pension milti hai?"}),
    ("pn07", "Pradhan Mantri Jan Dhan Yojana", {
        "en": "What is the overdraft limit on a Jan Dhan account?",
        "hi": "जन धन खाते पर ओवरड्राफ्ट सीमा कितनी है?",
        "hinglish": "Jan Dhan account par overdraft limit kitni hai?"}),
    ("pn08", "MGNREGA", {
        "en": "How many days of guaranteed work does MGNREGA provide in a year?",
        "hi": "मनरेगा में साल में कितने दिन के काम की गारंटी है?",
        "hinglish": "MGNREGA mein saal mein kitne din kaam ki guarantee hai?"}),
    ("pn09", "PM Garib Kalyan Anna Yojana", {
        "en": "How many kilograms of free grain are given under PM Garib Kalyan Anna Yojana?",
        "hi": "पीएम गरीब कल्याण अन्न योजना में कितने किलो मुफ्त अनाज मिलता है?",
        "hinglish": "PM Garib Kalyan Anna Yojana mein kitne kilo free ration milta hai?"}),
    ("pn10", "National Pension System", {
        "en": "What is the minimum annual contribution for a National Pension System account?",
        "hi": "राष्ट्रीय पेंशन प्रणाली खाते में न्यूनतम वार्षिक योगदान कितना है?",
        "hinglish": "NPS account mein saal ka minimum contribution kitna hai?"}),
    ("pn11", "PM Surya Ghar Muft Bijli Yojana", {
        "en": "How much subsidy is given for rooftop solar under PM Surya Ghar Muft Bijli Yojana?",
        "hi": "पीएम सूर्य घर मुफ्त बिजली योजना में रूफटॉप सोलर पर कितनी सब्सिडी मिलती है?",
        "hinglish": "PM Surya Ghar yojana mein rooftop solar par kitni subsidy milti hai?"}),
    ("pn12", "PM-KUSUM", {
        "en": "What subsidy does a farmer get for a solar pump under PM-KUSUM?",
        "hi": "पीएम-कुसुम में किसान को सोलर पंप पर कितनी सब्सिडी मिलती है?",
        "hinglish": "PM KUSUM mein solar pump par kitni subsidy milti hai?"}),
    ("pn13", "Swachh Bharat Mission (Gramin)", {
        "en": "How much incentive is given to build a household toilet under Swachh Bharat Mission?",
        "hi": "स्वच्छ भारत मिशन में घर में शौचालय बनाने के लिए कितनी प्रोत्साहन राशि मिलती है?",
        "hinglish": "Swachh Bharat Mission mein toilet banane ke liye kitna paisa milta hai?"}),
    ("pn14", "Senior Citizens Savings Scheme", {
        "en": "What is the maximum deposit allowed in the Senior Citizens Savings Scheme?",
        "hi": "वरिष्ठ नागरिक बचत योजना में अधिकतम कितनी राशि जमा कर सकते हैं?",
        "hinglish": "Senior Citizens Savings Scheme mein maximum kitna deposit kar sakte hain?"}),
    ("pn15", "PM Internship Scheme", {
        "en": "How much monthly stipend do interns get under the PM Internship Scheme?",
        "hi": "पीएम इंटर्नशिप योजना में इंटर्न को हर महीने कितना स्टाइपेंड मिलता है?",
        "hinglish": "PM Internship Scheme mein interns ko monthly kitna stipend milta hai?"}),
]

# `absent`: case-insensitive regexes run on the scheme's BODY (front matter
# excluded). If any matches, the source might state the answer and the build fails.
TRICK: list[tuple[str, str, list[str], dict[str, str]]] = [
    ("pt01", "pm_kisan", [r"helpline", r"toll.?free", r"\b1800\b", r"155261"], {
        "en": "What is the PM-KISAN helpline number?",
        "hi": "पीएम किसान का हेल्पलाइन नंबर क्या है?",
        "hinglish": "PM Kisan ka helpline number kya hai?"}),
    ("pt02", "pm_kisan", [r"\b\d{9,18}\b"], {
        "en": "What is the exact bank account number I should use to receive PM-KISAN money?",
        "hi": "पीएम किसान का पैसा पाने के लिए मुझे कौन सा बैंक खाता नंबर देना चाहिए?",
        "hinglish": "PM Kisan ka paisa lene ke liye kaunsa bank account number dena hai?"}),
    ("pt03", "pmjjby", [r"\bclaims?\b", r"settle"], {
        "en": "Within how many days is a PMJJBY death claim settled?",
        "hi": "पीएमजेजेबीवाई में मृत्यु दावे का निपटारा कितने दिनों में होता है?",
        "hinglish": "PMJJBY mein death claim kitne din mein settle hota hai?"}),
    ("pt04", "atal_pension_yojana", [r"rate of return", r"interest rate", r"\d+(\.\d+)?\s*%"], {
        "en": "What annual rate of return does Atal Pension Yojana pay on contributions?",
        "hi": "अटल पेंशन योजना योगदान पर सालाना कितना रिटर्न देती है?",
        "hinglish": "Atal Pension Yojana contribution par saal ka kitna return deti hai?"}),
    ("pt05", "pm_svanidhi", [r"\blakh", r"\bcrore", r"so far", r"till date"], {
        "en": "How many street vendors have received PM SVANidhi loans so far?",
        "hi": "अब तक कितने रेहड़ी-पटरी वालों को पीएम स्वनिधि लोन मिल चुका है?",
        "hinglish": "Ab tak kitne street vendors ko PM SVANidhi loan mil chuka hai?"}),
    ("pt06", "nmmss", [r"\b2026\b", r"exam date", r"will be held on"], {
        "en": "On what date will the NMMSS selection exam be held in 2026?",
        "hi": "2026 में एनएमएमएसएस चयन परीक्षा किस तारीख को होगी?",
        "hinglish": "2026 mein NMMSS exam kis date ko hoga?"}),
    ("pt07", "pm_fasal_bima", [r"hectare", r"punjab", r"wheat"], {
        "en": "How many rupees per hectare does a wheat farmer in Punjab pay as PMFBY premium?",
        "hi": "पंजाब में गेहूं के किसान को पीएमएफबीवाई में प्रति हेक्टेयर कितने रुपये प्रीमियम देना होता है?",
        "hinglish": "Punjab mein gehun ke kisan ko PMFBY mein per hectare kitna premium dena hota hai?"}),
    ("pt08", "pm_ujjwala", [r"subsid", r"per refill", r"\b300\b"], {
        "en": "What is the subsidy amount per LPG refill under Ujjwala?",
        "hi": "उज्ज्वला योजना में हर एलपीजी रिफिल पर कितनी सब्सिडी मिलती है?",
        "hinglish": "Ujjwala mein har LPG refill par kitni subsidy milti hai?"}),
    ("pt09", "stand_up_india", [r"interest"], {
        "en": "What is the interest rate on a Stand-Up India loan?",
        "hi": "स्टैंड-अप इंडिया लोन पर ब्याज दर क्या है?",
        "hinglish": "Stand-Up India loan ka interest rate kya hai?"}),
    ("pt10", "pmkvy_stt", [r"₹", r"\brs\.?\s*\d", r"stipend of"], {
        "en": "How much is the post-placement stipend under PMKVY per month?",
        "hi": "पीएमकेवीवाई में प्लेसमेंट के बाद हर महीने कितना स्टाइपेंड मिलता है?",
        "hinglish": "PMKVY mein placement ke baad monthly kitna stipend milta hai?"}),
    ("pt11", "pmay_urban", [r"last date", r"\b2026\b", r"deadline"], {
        "en": "What is the last date to apply for PMAY Urban in 2026?",
        "hi": "2026 में पीएमएवाई शहरी के लिए आवेदन की अंतिम तिथि क्या है?",
        "hinglish": "2026 mein PMAY Urban apply karne ki last date kya hai?"}),
    ("pt12", "kisan_credit_card", [r"\bacres?\b", r"\bhectares?\b"], {
        "en": "What is the exact KCC limit in rupees for a farmer with 2 acres of land?",
        "hi": "2 एकड़ ज़मीन वाले किसान को किसान क्रेडिट कार्ड पर ठीक कितने रुपये की सीमा मिलती है?",
        "hinglish": "2 acre zameen wale kisan ko KCC par exactly kitni limit milti hai?"}),
    ("pt13", "pm_vishwakarma", [r"\blakh artisans", r"\bcrore", r"so far", r"registered till"], {
        "en": "How many artisans have registered under PM Vishwakarma so far?",
        "hi": "अब तक पीएम विश्वकर्मा में कितने कारीगरों ने पंजीकरण कराया है?",
        "hinglish": "Ab tak PM Vishwakarma mein kitne karigaron ne registration kiya hai?"}),
]


def scheme_body(path: Path) -> str:
    """The markdown body without the YAML front matter (titles there are not facts)."""
    text = io.open(path, encoding="utf-8").read()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:]
    return text


def scheme_aliases(path: Path) -> set[str]:
    """Lower-cased name and short_name of an indexed scheme, from its front matter."""
    text = io.open(path, encoding="utf-8").read()
    names = set()
    for key in ("name", "short_name"):
        match = re.search(rf'^{key}:\s*"?(.*?)"?\s*$', text, flags=re.MULTILINE)
        if match:
            names.add(match.group(1).strip().lower())
    return names


def _check_languages(pair_id: str, texts: dict[str, str], problems: list[str]) -> None:
    if set(texts) != set(LANGUAGES) or not all(texts[lang].strip() for lang in LANGUAGES):
        problems.append(f"{pair_id}: needs exactly one non-empty question in each of {LANGUAGES}")


def build() -> tuple[list[dict], list[str]]:
    """Return (rows, problems). Rows are only valid when problems is empty."""
    paths = {path.stem: path for path in sorted(SCHEMES_DIR.glob("*.md"))}
    documents = {stem: normalize(io.open(p, encoding="utf-8").read()) for stem, p in paths.items()}
    bodies = {stem: scheme_body(p) for stem, p in paths.items()}
    indexed_names = set().union(*(scheme_aliases(p) for p in paths.values())) if paths else set()

    rows: list[dict] = []
    problems: list[str] = []
    seen_pairs: set[str] = set()

    def add(pair_id: str, texts: dict[str, str], **fields) -> None:
        if pair_id in seen_pairs:
            problems.append(f"{pair_id}: duplicate pair_id")
        seen_pairs.add(pair_id)
        _check_languages(pair_id, texts, problems)
        for lang in LANGUAGES:
            rows.append({"id": f"{pair_id}-{lang}", "pair_id": pair_id,
                         "question": texts.get(lang, ""), "language": lang, **fields})

    for pair_id, scheme, facts, category, texts in ANSWERABLE:
        if scheme not in documents:
            problems.append(f"{pair_id}: unknown scheme {scheme!r}")
        else:
            for fact in facts:
                if not fact_present(fact, documents[scheme]):
                    problems.append(f"{pair_id}: key_fact {fact!r} is NOT in {scheme}.md")
        add(pair_id, texts, answerable=True, expected_scheme=scheme,
            key_facts=facts, category=category)

    for pair_id, texts in OFF_TOPIC:
        add(pair_id, texts, answerable=False, expected_scheme=None,
            key_facts=[], category="off_topic")

    for pair_id, texts in FAKE_SCHEME:
        add(pair_id, texts, answerable=False, expected_scheme=None,
            key_facts=[], category="fake_scheme")

    for pair_id, real_scheme, texts in NOT_INDEXED:
        if real_scheme.lower() in indexed_names:
            problems.append(f"{pair_id}: {real_scheme!r} IS indexed — not a not_indexed question")
        add(pair_id, texts, answerable=False, expected_scheme=None, key_facts=[],
            category="not_indexed", real_scheme=real_scheme)

    for pair_id, scheme, absent, texts in TRICK:
        if scheme not in bodies:
            problems.append(f"{pair_id}: unknown scheme {scheme!r}")
        else:
            for pattern in absent:
                hit = re.search(pattern, bodies[scheme], flags=re.IGNORECASE)
                if hit:
                    problems.append(
                        f"{pair_id}: absent-pattern {pattern!r} MATCHES {scheme}.md "
                        f"({hit.group(0)!r}) — the source may state the answer"
                    )
        # expected_scheme stays None: no answer exists, so there is no "right"
        # scheme to retrieve. `target_scheme` records which one the question is about.
        add(pair_id, texts, answerable=False, expected_scheme=None, key_facts=[],
            category="trick_unstated_fact", target_scheme=scheme, absent_patterns=absent)

    return rows, problems


def main() -> int:
    rows, problems = build()
    if problems:
        print(f"FAILED: {len(problems)} problem(s) — the paper set was NOT written.\n")
        for problem in problems:
            print(f"  ! {problem}")
        return 1

    with io.open(OUT_PATH, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    answerable = sum(row["answerable"] for row in rows)
    by_category: dict[str, int] = {}
    for row in rows:
        by_category[row["category"] if not row["answerable"] else "answerable"] = (
            by_category.get(row["category"] if not row["answerable"] else "answerable", 0) + 1)
    print(f"Wrote {OUT_PATH} — {len(rows)} questions in {len(rows) // 3} translation-paired "
          f"triples ({answerable} answerable, {len(rows) - answerable} unanswerable)")
    for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {category}")
    print("\n  Checked: key facts present; trick facts absent; not-indexed schemes not indexed;"
          "\n  one question per language per pair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
