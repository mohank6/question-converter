import json
from openai import OpenAI
import logging

log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
log = logging.getLogger()
log.setLevel(logging.INFO)

# SYSTEM_PROMPT = f"""
#                 You are an experienced upsc faculty from a renowned coaching institute from delhi.\
#                 You have experience of converting upsc prelims exam mcqs from english to hindi in proper upsc context.\
#                 Based on knowledge of subjects like geography, history, environment, indian polity etc, convert following question from english to hindi.\
#                 Ensure to use technical terms and don't just convert from plain english to hindi.
#                 Input will be in json format:
#                 `statement` : Question statement
#                 `hint`: Hint for the question

#                 Respond in json format with keys:
#                 `statement` : Question converted from english to hindi
#                 `hint`: The corresponding converted hint
#                 **IMPORTANT RESPOND IN JSON FORMAT with two keys `statement` and `hint`**
#                 """
SYSTEM_PROMPT = """
**You are a UPSC prelims question expert specializing in converting English MCQs to Hindi.**

**Given a question in English (statement) and a hint, convert it to Hindi (statement) suitable for the UPSC prelims exam, maintaining technical accuracy and UPSC context.**

**Input (JSON):**

* `statement`: Question statement with options in English.
* `hint`: Hint for the question in English.

**Output (JSON):**

* `statement`: Converted question statement with options in Hindi.
* `hint`: Hint for the question in Hindi.

**IMPORTANT INFORMATION**
- DONOT change the base statement format. The answers number should remain the same in the converted statement.
- Maintain line breaks `\\n` in response similar to input
- DONOT forget to add multiple choices from end of each input `statement` to reponse `statement`.
"""
input_data = [
    '''
    "statement": "Consider  following statements regarding the representation of States in the Parliament: \n\n1. Delimitation of Constituencies is undertaken on the basis of census exercise to ensure that every State is represented in proportion to its population in both the Houses of Parliament.\n2. Delimitation Commission is a constitutional body, the notification of whose orders cannot be challenged in a Court. \n3. Territorial constituencies in States, at present, are based on the data of 2001 census, as the Constitution (87thAmendment) Act, 2003 enabled the delimitation exercise on the basis of 2001Census figures. \n4. As it stands today, Constitution of India prohibits any delimitation exercise till 2031. \nWhich of the statements given above are not correct   ?\n\n(A) 1, 2 and 4only\n(B) 2, 3 and 4only\n(C) 1, 3 and 4only\n(D) 1, 2, 3 and4\n",\
    "hint": "Delimitation constituencies are NOT applicable to representation of states in Council of States. Though it is correct to say the Order of delimitation commission, once notified, cannot be challenged in any Court, Delimitation commission is NOT a constitutional body but a statutory body. The Constitution has prohibited the revision of representation of States in the Lok Sabha till 2026, but not the delimitation of the Lok Sabha and Assembly constituencies...",
    ''',
    '''
    "statement": "Which of the following statements is/are true \nabout the Gram Sabha?  \n 1. All people living in a village or a group \nof villages are members of the Gram \nSabha. \n 2. All the plans for work of Gram Panchayat \nhave to be approved by Gram Sabha. \n 3. For better implementation of some \nspecific tasks, Gram Sabha form \ncommittees.  \n 4. The elected Secretary of the Gram Sabha \ncalls the meeting and keeps a record of \nthe proceedings.\n\n(A) 2 and 3\n(B) 1, 3 and 4\n(C) 2, 3 and 4\n(D) 1,2,3,4 ", \
    "hint": "Only adult villagers who have the right to \nvote can be member of Gram Sabha. Persons \nbelow 18 years of age can't become members. \n Gram Sabha plays a supervisory and \nmonitoring role over Gram Panchayat by \napproving it plan of work. \n Gram Sabha form committees like \nconstruction, animal husbandry, etc to carry \nout some specific tasks.  \n The Gram Panchayat has a Secretary who is \nalso the Secretary of the Gram Sabha. This \nperson is not an elected person but is \nappointed by the government. The Secretary \nis responsible for calling the meeting of the \nGram Sabha and Gram Panchayat and \nkeeping a record of the proceedings.",
    ''',
    '''
    "statement": "To which of the following the original jurisdiction of the supreme court does not \nextend? \n1. Inter-state water disputes. \n2. Matters referred to the Finance Commission. \n3. A dispute arising out of any pre-Constitution treaty, agreement, covenant, \nengagement, Sanad or other similar instrument. \nSelect the correct answer using the codes given below.\n\n(A) 1, 2 and 3 only\n(B) 2 and 3 only\n(C) 1 and 3 only\n(D) 1 and 2 only\n",  \
    "hint": " \nAs a federal court, the Supreme Court decides the disputes between different units of the \nIndian Federation. More elaborately, any dispute between: \n the Centre and one or more states; or \n the Centre and any state or states on one side and one or more states on the other;  \n or between two or more states.  \nIn the above federal disputes, the Supreme Court has exclusive original jurisdiction. \nExclusive means, no other court can decide such disputes and original means, the power to \nhear such disputes in the first instance, not by way of appeal. \nFurther, this jurisdiction of the Supreme Court does not extend to the following: \n A dispute arising out of any pre-Constitution treaty, agreement, covenant, \nengagement, Sanad or other similar instrument. \n A dispute arising out of any treaty, agreement, etc., which specifically provides that \nthe said jurisdiction does not extend to such a dispute. \n Inter-state water disputes. \n Matters referred to the Finance Commission. \n Adjustment of certain expenses and pensions between the Centre and the states. \n Ordinary dispute of Commercial nature between the Centre and the states. \n Recovery of damages by a state against the Centre.",
    ''',
    '''
    "statement": "Consider the following statements: \n1. Model Code of Conduct was issued for \nfirst general election after independence \nfor the first time. \n2. The Model Code of Conduct comes into \nforce immediately on announcement of \nthe election schedule by the Election \nCommission. \n3. The Parliament may make provision \nwith respect to all matters relating to \nelections to the Parliament and the state \nlegislatures including the model code of \nconduct. \nWhich of the statements given above is / \nare correct?\n\n(A) 2 only\n(B) 2 and 3 only\n(C) 1 and 2 only\n(D) 1, 2 and 3\n", \
    "hint": "Statement 1 is incorrect.  \nThe Code was issued for the first time in 1971 before the 5th Lok Sabha \nelections. Since then, it has been issued before every central and state election \nand revised from time to time. \nStatement 2 is correct.  \nThe Model Code of Conduct comes into force immediately on announcement of \nthe election schedule by the commission for the need of ensuring free and fair \nelections. \nStatement 3 is incorrect.  \nElection Commission of India's Model Code of Conduct is a set of guidelines \nissued by the Election Commission of India for conduct of political parties and \ncandidates during elections mainly with respect to speeches, polling day, \npolling booths, election manifestos, processions and general conduct. These set \nof norms has been evolved with the consensus of political parties who have \nconsented to abide by the principles embodied in the said code in its letter and \nspirit.",
    ''',
    '''
    "statement": "Centre has announced a recapitalization plan for the Public-Sector Banks (PSBs) \nthrough issuance of recapitalization bonds. Consider the benefits of recap bonds. \n1. There will be less burden on taxpayer. \n2. Government can avoid crowding out private borrowings. \n3. The method is potential solution for the structural problems in the banking system. \nWhich of the above statements is/are correct?\n\n(A) 1, 2 and 3\n(B) 1 and 3 only\n(C) 2 and 3 only\n(D) 1 and 2 only\n", \
    "hint": "The recapitalization plan is a three-part package: Rs. 18000 crores from the budget, Rs. \n58000 crores that banks can raise by diluting their equity and Rs. 1.35 lakh crore through \nissuance of recap bonds. \nRecapitalization Bonds approach \n It refers to using equity money in order to restructure an institution’s debt. \n The bonds can be issued either directly by the government or through a holding \ncompany. \n The government will issue bonds to the banks for a share of the bank’s Equity. \n The annual interest on these bonds and the principal on redemption will be paid by \nthe central government. \n These bonds can be sold off by the banks in the market when in need of capita \nBenefits of Recapitalization Bonds \n The government need not to raise immediate tax revenues to fund the mounting bill \non bank recapitalization, which means less burden on the taxpayer. \n Borrowing directly from the banking system instead of the markets, the government \ncan avoid crowding out private borrowings or distorting market yields. \n Recapitalization Bonds does not strain the banking finances, because lending to the \ngovernment is safest for their loan funds. In any case, public sector banks tend to \ninvest well in excess of their Statutory Liquidity Ratio requirements in government \nsecurities. \nLimitation of Recapitalization Bonds \n The method is not the solution for the structural problems in the banking system \nthat have been created by the bad loan menace, poor governance systems, badly \njudged lending decisions, and the repeated overlooking of doubtful accounts of \npotential NPAs. The nature of capital infusion shows that it is a kind of bailout \noffering and not necessarily trying to aid the banks in growth. \n The credit demand of loans is weak in the current market, which could have negative \nimpact on banking operation.",
    ''',

]
reponse_data = []
for i in input_data:
    USER_PROMPT = f"""
                    {
                    {i.strip()}
                    }
"""
    data = OpenAI.generate_completion(SYSTEM_PROMPT, USER_PROMPT)
    reponse_data.append(data)

with open('response.json', 'w') as fp:
    json.dump(reponse_data, fp)
