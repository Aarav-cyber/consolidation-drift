"""Expand the 20-item atomic seed (atomic_v0) to a 200-item balanced corpus.

Keeps all 20 v0 rows verbatim. Adds 180 synthetic rows so each category
reaches 40 items with true_level mass centered at 3 (drift can move either
way without immediate ceiling/floor clipping):

  pref/cert/obl : L1:4 L2:8 L3:12 L4:8 L5:8
  num           : L1:4 L2:8 L3:12 L4:8 L5:8  (v0 has L2x2,L3x2 -> adds differ)
  neg           : 40 binary items (negation scope, true/false underlying value)

Every row keeps the v0 schema: id, category, original_text, attribute,
true_level (graded only), true_value/unit (numeric + negation), scale,
source, source_reference.

Deterministic output (fixed order, no RNG) so diffs are reviewable.
"""
import json
from pathlib import Path

V0_PATH = Path("data/atomic/atomic_v0.jsonl")
V1_PATH = Path("data/atomic/atomic_v1.jsonl")
SRC = "Synthetic DriftBench v1; template family from atomic_v0 + LongMemEval single-session-preference pattern."
REF = "Project proposal; LongMemEval"

# ---------------------------------------------------------------- pref ---
# scale convention (matches v0): 1 = opposite pole, 2 = slight, 3 = prefer,
# 4 = strong, 5 = requires. Each item carries its own 5-point scale.
PREF_NEW = [
    # ---- level 1 (opposite pole) : need 4
    ("evening_workout_preference", 1,
     "I strongly prefer evening workouts; morning sessions do not suit me at all.",
     {"1": "strongly prefers evening", "2": "slightly prefers morning", "3": "prefers morning", "4": "strongly prefers morning", "5": "requires morning"}),
    ("window_seat_preference", 1,
     "I strongly prefer the aisle seat and dislike sitting by the window.",
     {"1": "strongly prefers aisle", "2": "slightly prefers window", "3": "prefers window", "4": "strongly prefers window", "5": "requires window seat"}),
    ("dark_mode_preference", 1,
     "I strongly prefer light mode and find dark mode hard to read.",
     {"1": "strongly prefers light mode", "2": "slightly prefers dark mode", "3": "prefers dark mode", "4": "strongly prefers dark mode", "5": "requires dark mode"}),
    ("phone_call_preference", 1,
     "I strongly prefer texting; I avoid phone calls whenever possible.",
     {"1": "strongly prefers texting", "2": "slightly prefers calls", "3": "prefers calls", "4": "strongly prefers calls", "5": "requires phone calls"}),
    # ---- level 2 (slight) : need 7
    ("commute_mode_preference", 2,
     "I slightly prefer cycling to work, but driving is fine too.",
     {"1": "strongly prefers driving", "2": "slightly prefers cycling", "3": "prefers cycling", "4": "strongly prefers cycling", "5": "requires cycling"}),
    ("lunch_spot_preference", 2,
     "I slightly prefer the canteen, though the cafe nearby is fine as well.",
     {"1": "strongly prefers cafe", "2": "slightly prefers canteen", "3": "prefers canteen", "4": "strongly prefers canteen", "5": "requires canteen"}),
    ("music_genre_preference", 2,
     "I slightly prefer jazz over rock, but I enjoy rock too.",
     {"1": "strongly prefers rock", "2": "slightly prefers jazz", "3": "prefers jazz", "4": "strongly prefers jazz", "5": "only listens to jazz"}),
    ("note_tool_preference", 2,
     "I slightly prefer paper notes, although digital notes work for me.",
     {"1": "strongly prefers digital", "2": "slightly prefers paper", "3": "prefers paper", "4": "strongly prefers paper", "5": "requires paper notes"}),
    ("morning_standup_preference", 2,
     "I slightly prefer standups at 10 AM, but 11 AM is acceptable.",
     {"1": "strongly prefers afternoon", "2": "slightly prefers 10 AM", "3": "prefers 10 AM", "4": "strongly prefers 10 AM", "5": "requires 10 AM"}),
    ("coffee_strength_preference", 2,
     "I slightly prefer mild coffee, though strong coffee is okay occasionally.",
     {"1": "strongly prefers strong", "2": "slightly prefers mild", "3": "prefers mild", "4": "strongly prefers mild", "5": "only drinks mild"}),
    ("desk_location_preference", 2,
     "I slightly prefer sitting near the window, but I can sit anywhere.",
     {"1": "strongly prefers interior", "2": "slightly prefers window", "3": "prefers window", "4": "strongly prefers window", "5": "requires window desk"}),
    # ---- level 3 (moderate) : need 11
    ("breakfast_food_preference", 3,
     "I prefer idli for breakfast over dosa, though I eat both.",
     {"1": "strongly prefers dosa", "2": "slightly prefers idli", "3": "prefers idli", "4": "strongly prefers idli", "5": "only eats idli"}),
    ("editor_theme_preference", 3,
     "I prefer a dark editor theme over a light one for daily coding.",
     {"1": "strongly prefers light", "2": "slightly prefers dark", "3": "prefers dark", "4": "strongly prefers dark", "5": "requires dark theme"}),
    ("weekend_activity_preference", 3,
     "I prefer hiking on weekends rather than visiting malls.",
     {"1": "strongly prefers malls", "2": "slightly prefers hiking", "3": "prefers hiking", "4": "strongly prefers hiking", "5": "only hikes"}),
    ("payment_mode_preference", 3,
     "I prefer paying by card rather than cash for most purchases.",
     {"1": "strongly prefers cash", "2": "slightly prefers card", "3": "prefers card", "4": "strongly prefers card", "5": "only pays by card"}),
    ("reading_format_preference", 3,
     "I prefer physical books over ebooks for novels.",
     {"1": "strongly prefers ebooks", "2": "slightly prefers physical", "3": "prefers physical", "4": "strongly prefers physical", "5": "only reads physical"}),
    ("meeting_length_preference", 3,
     "I prefer 30-minute meetings over hour-long ones.",
     {"1": "strongly prefers hour-long", "2": "slightly prefers 30-minute", "3": "prefers 30-minute", "4": "strongly prefers 30-minute", "5": "requires 30-minute"}),
    ("keyboard_layout_preference", 3,
     "I prefer a mechanical keyboard over a membrane one for typing.",
     {"1": "strongly prefers membrane", "2": "slightly prefers mechanical", "3": "prefers mechanical", "4": "strongly prefers mechanical", "5": "only uses mechanical"}),
    ("travel_seat_preference", 3,
     "I prefer the upper berth on overnight trains.",
     {"1": "strongly prefers lower berth", "2": "slightly prefers upper", "3": "prefers upper berth", "4": "strongly prefers upper", "5": "requires upper berth"}),
    ("tea_flavor_preference", 3,
     "I prefer ginger tea over plain tea in the evenings.",
     {"1": "strongly prefers plain", "2": "slightly prefers ginger", "3": "prefers ginger tea", "4": "strongly prefers ginger", "5": "only drinks ginger tea"}),
    ("code_review_tool_preference", 3,
     "I prefer reviewing code in the morning rather than late evening.",
     {"1": "strongly prefers evening review", "2": "slightly prefers morning", "3": "prefers morning review", "4": "strongly prefers morning", "5": "only reviews in morning"}),
    ("office_snack_preference", 3,
     "I prefer fruit over biscuits as an office snack.",
     {"1": "strongly prefers biscuits", "2": "slightly prefers fruit", "3": "prefers fruit", "4": "strongly prefers fruit", "5": "only eats fruit"}),
    # ---- level 4 (strong) : need 7
    ("remote_work_preference", 4,
     "I strongly prefer working remotely and rarely come to the office.",
     {"1": "strongly prefers office", "2": "slightly prefers remote", "3": "prefers remote", "4": "strongly prefers remote", "5": "requires remote"}),
    ("morning_gym_preference", 4,
     "I strongly prefer morning gym sessions; evening sessions feel wrong.",
     {"1": "strongly prefers evening gym", "2": "slightly prefers morning", "3": "prefers morning", "4": "strongly prefers morning", "5": "requires morning gym"}),
    ("veg_food_preference", 4,
     "I strongly prefer vegetarian meals and seldom eat non-vegetarian food.",
     {"1": "prefers non-vegetarian", "2": "slightly prefers vegetarian", "3": "prefers vegetarian", "4": "strongly prefers vegetarian", "5": "requires vegetarian"}),
    ("async_communication_preference", 4,
     "I strongly prefer async updates over live meetings for status reports.",
     {"1": "strongly prefers live meetings", "2": "slightly prefers async", "3": "prefers async", "4": "strongly prefers async", "5": "requires async"}),
    ("hindi_movie_preference", 4,
     "I strongly prefer Hindi films over dubbed versions for weekend viewing.",
     {"1": "strongly prefers dubbed", "2": "slightly prefers Hindi", "3": "prefers Hindi", "4": "strongly prefers Hindi", "5": "only watches Hindi"}),
    ("standing_desk_preference", 4,
     "I strongly prefer my standing desk and rarely sit during work hours.",
     {"1": "strongly prefers sitting", "2": "slightly prefers standing", "3": "prefers standing", "4": "strongly prefers standing", "5": "requires standing desk"}),
    ("local_train_preference", 4,
     "I strongly prefer the local train over the bus for my daily commute.",
     {"1": "strongly prefers bus", "2": "slightly prefers train", "3": "prefers train", "4": "strongly prefers train", "5": "only takes train"}),
    # ---- level 5 (requires) : need 7
    ("allergy_diet_requirement", 5,
     "I require gluten-free meals at all work events for medical reasons.",
     {"1": "prefers regular food", "2": "slightly prefers gluten-free", "3": "prefers gluten-free", "4": "strongly prefers gluten-free", "5": "requires gluten-free"}),
    ("wheelchair_access_requirement", 5,
     "I require step-free building access and cannot use stairs.",
     {"1": "prefers stairs", "2": "slightly prefers step-free", "3": "prefers step-free", "4": "strongly prefers step-free", "5": "requires step-free access"}),
    ("sign_language_requirement", 5,
     "I require sign-language interpretation for all-hands meetings.",
     {"1": "no interpretation needed", "2": "slightly prefers interpretation", "3": "prefers interpretation", "4": "strongly prefers interpretation", "5": "requires interpretation"}),
    ("vegan_meal_requirement", 5,
     "I require fully vegan catering with no dairy or eggs.",
     {"1": "prefers regular catering", "2": "slightly prefers vegan", "3": "prefers vegan", "4": "strongly prefers vegan", "5": "requires vegan"}),
    ("large_font_requirement", 5,
     "I require large-font slides in every presentation I attend.",
     {"1": "prefers standard font", "2": "slightly prefers large font", "3": "prefers large font", "4": "strongly prefers large font", "5": "requires large font"}),
    ("quiet_room_requirement", 5,
     "I require a quiet room for focused work and cannot work in open halls.",
     {"1": "prefers open hall", "2": "slightly prefers quiet", "3": "prefers quiet room", "4": "strongly prefers quiet", "5": "requires quiet room"}),
    ("morning_shift_requirement", 5,
     "I require the morning shift and cannot take night shifts.",
     {"1": "prefers night shift", "2": "slightly prefers morning", "3": "prefers morning", "4": "strongly prefers morning", "5": "requires morning shift"}),
]

# ---------------------------------------------------------------- cert ---
CERT_NEW = [
    # ---- level 1 (disbelieve) : need 4
    ("leave_approval_belief", 1,
     "I am certain my leave request was not approved yet.",
     {"1": "certainly not approved", "2": "possibly approved", "3": "probably approved", "4": "very likely approved", "5": "certainly approved"}),
    ("bus_on_time_belief", 1,
     "I am sure the bus has already left without me.",
     {"1": "certainly missed it", "2": "might still catch it", "3": "probably on time", "4": "very likely on time", "5": "certainly on time"}),
    ("server_online_belief", 1,
     "I am certain the staging server is still offline.",
     {"1": "certainly offline", "2": "possibly online", "3": "probably online", "4": "very likely online", "5": "certainly online"}),
    ("rain_today_belief", 1,
     "I am sure it will not rain today despite the clouds.",
     {"1": "certainly no rain", "2": "might rain", "3": "probably rains", "4": "very likely rains", "5": "certainly rains"}),
    # ---- level 2 (might) : need 7
    ("cab_arrival_belief", 2,
     "The cab might reach in ten minutes, but traffic looks bad.",
     {"1": "certainly will not arrive", "2": "might arrive", "3": "probably will arrive", "4": "almost certainly will arrive", "5": "definitely will arrive"}),
    ("salary_credit_belief", 2,
     "My salary might be credited today, though it is usually late.",
     {"1": "certainly not credited", "2": "might be credited", "3": "probably credited", "4": "very likely credited", "5": "certainly credited"}),
    ("guest_visit_belief", 2,
     "Our guests might come over this Sunday, but nothing is fixed.",
     {"1": "certainly not coming", "2": "might come", "3": "probably coming", "4": "very likely coming", "5": "certainly coming"}),
    ("exam_result_belief", 2,
     "I might have cleared the cutoff, but I am not certain.",
     {"1": "certainly not cleared", "2": "might have cleared", "3": "probably cleared", "4": "very likely cleared", "5": "certainly cleared"}),
    ("power_restore_belief", 2,
     "Power might be restored by evening, though the fault looks major.",
     {"1": "certainly not restored", "2": "might be restored", "3": "probably restored", "4": "very likely restored", "5": "certainly restored"}),
    ("offer_letter_belief", 2,
     "I might receive the offer letter this week, but HR is slow.",
     {"1": "certainly not this week", "2": "might arrive", "3": "probably arrives", "4": "very likely arrives", "5": "certainly arrives"}),
    ("match_ticket_belief", 2,
     "We might still get match tickets at the counter tomorrow.",
     {"1": "certainly sold out", "2": "might get tickets", "3": "probably get tickets", "4": "very likely get tickets", "5": "certainly get tickets"}),
    # ---- level 3 (probably) : need 11
    ("delivery_date_belief", 3,
     "The courier will probably deliver the parcel by Tuesday.",
     {"1": "certainly not by Tuesday", "2": "might deliver", "3": "probably delivers", "4": "very likely delivers", "5": "certainly delivers"}),
    ("interview_round_belief", 3,
     "I will probably clear this interview round given my preparation.",
     {"1": "certainly will not clear", "2": "might clear", "3": "probably clears", "4": "very likely clears", "5": "certainly clears"}),
    ("flight_delay_belief", 3,
     "Our flight will probably be delayed by an hour due to fog.",
     {"1": "certainly on time", "2": "might be delayed", "3": "probably delayed", "4": "very likely delayed", "5": "certainly delayed"}),
    ("festival_bonus_belief", 3,
     "The company will probably announce the festival bonus next week.",
     {"1": "certainly no bonus", "2": "might announce", "3": "probably announces", "4": "very likely announces", "5": "certainly announces"}),
    ("metro_extension_belief", 3,
     "The new metro line will probably open before December.",
     {"1": "certainly delayed", "2": "might open", "3": "probably opens", "4": "very likely opens", "5": "certainly opens"}),
    ("colleague_joining_belief", 3,
     "Our new colleague will probably join on Monday.",
     {"1": "certainly not Monday", "2": "might join", "3": "probably joins", "4": "very likely joins", "5": "certainly joins"}),
    ("water_supply_belief", 3,
     "The water supply will probably resume by morning.",
     {"1": "certainly not by morning", "2": "might resume", "3": "probably resumes", "4": "very likely resumes", "5": "certainly resumes"}),
    ("workshop_happening_belief", 3,
     "The weekend workshop will probably happen as scheduled.",
     {"1": "certainly cancelled", "2": "might happen", "3": "probably happens", "4": "very likely happens", "5": "certainly happens"}),
    ("refund_credit_belief", 3,
     "My refund will probably be credited within five days.",
     {"1": "certainly not credited", "2": "might be credited", "3": "probably credited", "4": "very likely credited", "5": "certainly credited"}),
    ("exam_schedule_belief", 3,
     "The exams will probably start in the first week of March.",
     {"1": "certainly postponed", "2": "might start", "3": "probably starts", "4": "very likely starts", "5": "certainly starts"}),
    ("playoff_qualification_belief", 3,
     "Our team will probably qualify for the playoffs this season.",
     {"1": "certainly knocked out", "2": "might qualify", "3": "probably qualifies", "4": "very likely qualifies", "5": "certainly qualifies"}),
    # ---- level 4 (very confident) : need 7
    ("project_demo_belief", 4,
     "I am fairly confident our demo will impress the client tomorrow.",
     {"1": "certainly unimpressed", "2": "possibly impressed", "3": "probably impressed", "4": "very confident of impressing", "5": "certain of impressing"}),
    ("rent_agreement_belief", 4,
     "I am fairly confident the rent agreement will be signed this week.",
     {"1": "certainly not signed", "2": "possibly signed", "3": "probably signed", "4": "very confident it is signed", "5": "certain it is signed"}),
    ("medical_report_belief", 4,
     "I am fairly confident my reports will come back normal.",
     {"1": "certainly abnormal", "2": "possibly normal", "3": "probably normal", "4": "very confident they are normal", "5": "certain they are normal"}),
    ("admission_list_belief", 4,
     "I am fairly confident my name is on the admission list.",
     {"1": "certainly not listed", "2": "possibly listed", "3": "probably listed", "4": "very confident of listing", "5": "certain of listing"}),
    ("visa_approval_belief", 4,
     "I am fairly confident my visa will be approved without an interview.",
     {"1": "certainly rejected", "2": "possibly approved", "3": "probably approved", "4": "very confident of approval", "5": "certain of approval"}),
    ("promotion_belief", 4,
     "I am fairly confident the promotion will be announced in April.",
     {"1": "certainly no promotion", "2": "possibly announced", "3": "probably announced", "4": "very confident of announcement", "5": "certain of announcement"}),
    ("wedding_date_belief", 4,
     "I am fairly confident the wedding is fixed for February 14.",
     {"1": "certainly not February", "2": "possibly February", "3": "probably February", "4": "very confident of February 14", "5": "certain of February 14"}),
    # ---- level 5 (certain) : need 7
    ("fee_payment_belief", 5,
     "I am certain the fees were paid before the deadline.",
     {"1": "certainly unpaid", "2": "possibly paid", "3": "probably paid", "4": "very likely paid", "5": "certainly paid"}),
    ("ticket_booking_belief", 5,
     "I am certain our train tickets are confirmed for Friday.",
     {"1": "certainly waitlisted", "2": "possibly confirmed", "3": "probably confirmed", "4": "very likely confirmed", "5": "certainly confirmed"}),
    ("degree_certificate_belief", 5,
     "I am certain my degree certificate arrived at the office.",
     {"1": "certainly not arrived", "2": "possibly arrived", "3": "probably arrived", "4": "very likely arrived", "5": "certainly arrived"}),
    ("otp_verified_belief", 5,
     "I am certain the OTP was verified on the first attempt.",
     {"1": "certainly failed", "2": "possibly verified", "3": "probably verified", "4": "very likely verified", "5": "certainly verified"}),
    ("attendance_marked_belief", 5,
     "I am certain my attendance was marked for all five days.",
     {"1": "certainly missing days", "2": "possibly marked", "3": "probably marked", "4": "very likely marked", "5": "certainly marked"}),
    ("passport_renewed_belief", 5,
     "I am certain my passport renewal was completed last month.",
     {"1": "certainly pending", "2": "possibly completed", "3": "probably completed", "4": "very likely completed", "5": "certainly completed"}),
    ("goal_scored_belief", 5,
     "I am certain our team scored in the final minute of the match.",
     {"1": "certainly no goal", "2": "possibly scored", "3": "probably scored", "4": "very likely scored", "5": "certainly scored"}),
]

# ------------------------------------------------------------------ obl ---
OBL_NEW = [
    # ---- level 1 (no obligation) : need 4
    ("gym_membership_obligation", 1,
     "I have no obligation to renew my gym membership this year.",
     {"1": "no obligation", "2": "should renew", "3": "need to renew", "4": "strongly need to renew", "5": "must renew"}),
    ("party_attendance_obligation", 1,
     "I have no obligation to attend the farewell party on Friday.",
     {"1": "no obligation", "2": "should attend", "3": "need to attend", "4": "strongly need to attend", "5": "must attend"}),
    ("donation_obligation", 1,
     "I have no obligation to donate to this fundraiser.",
     {"1": "no obligation", "2": "should donate", "3": "need to donate", "4": "strongly need to donate", "5": "must donate"}),
    ("overtime_obligation", 1,
     "I have no obligation to work this weekend.",
     {"1": "no obligation", "2": "should work", "3": "need to work", "4": "strongly need to work", "5": "must work"}),
    # ---- level 2 (should) : need 7
    ("dentist_visit_obligation", 2,
     "I should visit the dentist for a cleaning sometime soon.",
     {"1": "no obligation", "2": "should visit", "3": "need to visit", "4": "strongly need to visit", "5": "must visit"}),
    ("backup_laptop_obligation", 2,
     "I should back up my laptop before the trip, but it can wait.",
     {"1": "no obligation", "2": "should back up", "3": "need to back up", "4": "strongly need to back up", "5": "must back up"}),
    ("library_book_obligation", 2,
     "I should return the library books this week if I get time.",
     {"1": "no obligation", "2": "should return", "3": "need to return", "4": "strongly need to return", "5": "must return"}),
    ("water_plants_obligation", 2,
     "I should water the balcony plants before they dry out.",
     {"1": "no obligation", "2": "should water", "3": "need to water", "4": "strongly need to water", "5": "must water"}),
    ("update_resume_obligation", 2,
     "I should update my resume with the new project details soon.",
     {"1": "no obligation", "2": "should update", "3": "need to update", "4": "strongly need to update", "5": "must update"}),
    ("call_parents_obligation", 2,
     "I should call my parents this weekend, though nothing is urgent.",
     {"1": "no obligation", "2": "should call", "3": "need to call", "4": "strongly need to call", "5": "must call"}),
    ("clean_desk_obligation", 2,
     "I should clean my desk before the audit visit next month.",
     {"1": "no obligation", "2": "should clean", "3": "need to clean", "4": "strongly need to clean", "5": "must clean"}),
    # ---- level 3 (need) : need 11
    ("submit_timesheet_obligation", 3,
     "I need to submit my timesheet before Friday evening.",
     {"1": "no obligation", "2": "should submit", "3": "need to submit", "4": "strongly need to submit", "5": "must submit"}),
    ("pay_electricity_obligation", 3,
     "I need to pay the electricity bill before the due date.",
     {"1": "no obligation", "2": "should pay", "3": "need to pay", "4": "strongly need to pay", "5": "must pay immediately"}),
    ("collect_id_card_obligation", 3,
     "I need to collect my new ID card from the front desk.",
     {"1": "no obligation", "2": "should collect", "3": "need to collect", "4": "strongly need to collect", "5": "must collect"}),
    ("attend_training_obligation", 3,
     "I need to attend the safety training session on Thursday.",
     {"1": "no obligation", "2": "should attend", "3": "need to attend", "4": "strongly need to attend", "5": "must attend"}),
    ("file_tax_return_obligation", 3,
     "I need to file my tax return before the end of July.",
     {"1": "no obligation", "2": "should file", "3": "need to file", "4": "strongly need to file", "5": "must file"}),
    ("service_bike_obligation", 3,
     "I need to service my bike before the long ride next month.",
     {"1": "no obligation", "2": "should service", "3": "need to service", "4": "strongly need to service", "5": "must service"}),
    ("confirm_hotel_obligation", 3,
     "I need to confirm the hotel booking for the conference.",
     {"1": "no obligation", "2": "should confirm", "3": "need to confirm", "4": "strongly need to confirm", "5": "must confirm"}),
    ("verify_aadhaar_obligation", 3,
     "I need to verify my address proof for the new bank account.",
     {"1": "no obligation", "2": "should verify", "3": "need to verify", "4": "strongly need to verify", "5": "must verify"}),
    ("prepare_slides_obligation", 3,
     "I need to prepare the slides for Monday's review.",
     {"1": "no obligation", "2": "should prepare", "3": "need to prepare", "4": "strongly need to prepare", "5": "must prepare"}),
    ("recharge_metro_obligation", 3,
     "I need to recharge my metro card before Monday.",
     {"1": "no obligation", "2": "should recharge", "3": "need to recharge", "4": "strongly need to recharge", "5": "must recharge"}),
    ("inform_manager_obligation", 3,
     "I need to inform my manager about the client escalation today.",
     {"1": "no obligation", "2": "should inform", "3": "need to inform", "4": "strongly need to inform", "5": "must inform"}),
    # ---- level 4 (strongly need) : need 7
    ("fix_production_bug_obligation", 4,
     "I really need to fix the production bug before the demo.",
     {"1": "no obligation", "2": "should fix", "3": "need to fix", "4": "strongly need to fix", "5": "must fix"}),
    ("carry_passport_obligation", 4,
     "I really need to carry my passport for tomorrow's verification.",
     {"1": "no obligation", "2": "should carry", "3": "need to carry", "4": "strongly need to carry", "5": "must carry"}),
    ("lock_server_room_obligation", 4,
     "I really need to lock the server room before leaving tonight.",
     {"1": "no obligation", "2": "should lock", "3": "need to lock", "4": "strongly need to lock", "5": "must lock"}),
    ("inform_society_obligation", 4,
     "I really need to inform the society office about the water leakage.",
     {"1": "no obligation", "2": "should inform", "3": "need to inform", "4": "strongly need to inform", "5": "must inform"}),
    ("charge_phone_obligation", 4,
     "I really need to charge my phone before the long journey.",
     {"1": "no obligation", "2": "should charge", "3": "need to charge", "4": "strongly need to charge", "5": "must charge"}),
    ("submit_leave_obligation", 4,
     "I really need to submit my leave application before the roster freezes.",
     {"1": "no obligation", "2": "should submit", "3": "need to submit", "4": "strongly need to submit", "5": "must submit"}),
    ("save_presentation_obligation", 4,
     "I really need to save the presentation before the system update.",
     {"1": "no obligation", "2": "should save", "3": "need to save", "4": "strongly need to save", "5": "must save"}),
    # ---- level 5 (must) : need 7
    ("evacuate_building_obligation", 5,
     "I must evacuate the building immediately when the alarm sounds.",
     {"1": "no obligation", "2": "should evacuate", "3": "need to evacuate", "4": "strongly need to evacuate", "5": "must evacuate"}),
    ("wear_helmet_obligation", 5,
     "I must wear a helmet on every bike ride without exception.",
     {"1": "no obligation", "2": "should wear", "3": "need to wear", "4": "strongly need to wear", "5": "must wear"}),
    ("verify_payment_obligation", 5,
     "I must verify the payment link before entering my card details.",
     {"1": "no obligation", "2": "should verify", "3": "need to verify", "4": "strongly need to verify", "5": "must verify"}),
    ("carry_ticket_obligation", 5,
     "I must carry a valid ticket for the entire train journey.",
     {"1": "no obligation", "2": "should carry", "3": "need to carry", "4": "strongly need to carry", "5": "must carry"}),
    ("switch_off_gas_obligation", 5,
     "I must switch off the gas cylinder before leaving the house.",
     {"1": "no obligation", "2": "should switch off", "3": "need to switch off", "4": "strongly need to switch off", "5": "must switch off"}),
    ("report_incident_obligation", 5,
     "I must report any security incident to the team within an hour.",
     {"1": "no obligation", "2": "should report", "3": "need to report", "4": "strongly need to report", "5": "must report"}),
    ("backup_database_obligation", 5,
     "I must back up the production database before every migration.",
     {"1": "no obligation", "2": "should back up", "3": "need to back up", "4": "strongly need to back up", "5": "must back up"}),
]

# ------------------------------------------------------------------ num ---
# (attribute, level, text, true_value, unit, scale)
NUM_NEW = [
    # ---- level 1 : need 4
    ("printer_pages", 1, "We need to print about 5 pages for the handout.", 5, "pages",
     {"1": "5 pages", "2": "10 pages", "3": "20 pages", "4": "50 pages", "5": "100 pages"}),
    ("meeting_attendees", 1, "Only about 3 people are joining the review call.", 3, "people",
     {"1": "3 people", "2": "6 people", "3": "12 people", "4": "25 people", "5": "50 people"}),
    ("bus_fare", 1, "The bus fare is around 10 rupees per trip.", 10, "INR",
     {"1": "10 INR", "2": "25 INR", "3": "50 INR", "4": "100 INR", "5": "200 INR"}),
    ("walk_distance", 1, "My office is about 1 km from the station.", 1, "km",
     {"1": "1 km", "2": "3 km", "3": "8 km", "4": "15 km", "5": "30 km"}),
    # ---- level 2 : need 6 (v0 already has 2x L2 -> total L2 = 8)
    ("workshop_participants", 2, "Around 15 people registered for the weekend workshop.", 15, "people",
     {"1": "5 people", "2": "15 people", "3": "30 people", "4": "60 people", "5": "120 people"}),
    ("water_tank_capacity", 2, "Our building tank holds about 500 litres of water.", 500, "litres",
     {"1": "200 litres", "2": "500 litres", "3": "1000 litres", "4": "2000 litres", "5": "5000 litres"}),
    ("pen_drive_size", 2, "I carry a 16 GB pen drive for file transfers.", 16, "GB",
     {"1": "4 GB", "2": "16 GB", "3": "64 GB", "4": "256 GB", "5": "1 TB"}),
    ("exam_duration", 2, "The test lasts about 60 minutes in total.", 60, "minutes",
     {"1": "30 minutes", "2": "60 minutes", "3": "120 minutes", "4": "180 minutes", "5": "240 minutes"}),
    ("novel_pages", 2, "The novel I am reading has around 200 pages.", 200, "pages",
     {"1": "80 pages", "2": "200 pages", "3": "400 pages", "4": "700 pages", "5": "1000 pages"}),
    ("monthly_grocery_budget", 2, "Our monthly grocery budget is around 4000 rupees.", 4000, "INR",
     {"1": "1500 INR", "2": "4000 INR", "3": "8000 INR", "4": "15000 INR", "5": "30000 INR"}),
    # ---- level 3 : need 10 (v0 already has 2x L3 -> total L3 = 12)
    ("conference_attendees", 3, "About 150 people are expected at the conference.", 150, "people",
     {"1": "30 people", "2": "75 people", "3": "150 people", "4": "300 people", "5": "600 people"}),
    ("backup_size", 3, "The full backup is roughly 250 GB.", 250, "GB",
     {"1": "25 GB", "2": "100 GB", "3": "250 GB", "4": "500 GB", "5": "1 TB"}),
    ("loan_amount", 3, "I took a personal loan of about 200000 rupees.", 200000, "INR",
     {"1": "25000 INR", "2": "75000 INR", "3": "200000 INR", "4": "500000 INR", "5": "1000000 INR"}),
    ("marathon_distance", 3, "The marathon route is just over 21 km.", 21, "km",
     {"1": "5 km", "2": "10 km", "3": "21 km", "4": "42 km", "5": "80 km"}),
    ("notice_period", 3, "My notice period is 45 days.", 45, "days",
     {"1": "7 days", "2": "15 days", "3": "45 days", "4": "90 days", "5": "180 days"}),
    ("class_strength", 3, "There are 40 students in my daughter's class.", 40, "students",
     {"1": "10 students", "2": "20 students", "3": "40 students", "4": "80 students", "5": "150 students"}),
    ("flight_duration", 3, "The flight takes about 180 minutes gate to gate.", 180, "minutes",
     {"1": "45 minutes", "2": "90 minutes", "3": "180 minutes", "4": "360 minutes", "5": "720 minutes"}),
    ("wedding_guests", 3, "We invited around 300 guests to the wedding.", 300, "guests",
     {"1": "50 guests", "2": "120 guests", "3": "300 guests", "4": "600 guests", "5": "1200 guests"}),
    ("book_collection", 3, "I own about 120 books in total.", 120, "books",
     {"1": "15 books", "2": "50 books", "3": "120 books", "4": "300 books", "5": "800 books"}),
    ("plot_area", 3, "The plot measures roughly 1500 square feet.", 1500, "sqft",
     {"1": "300 sqft", "2": "800 sqft", "3": "1500 sqft", "4": "3000 sqft", "5": "6000 sqft"}),
    # ---- level 4 : need 8
    ("stadium_capacity", 4, "The stadium holds close to 40000 spectators.", 40000, "spectators",
     {"1": "2000 spectators", "2": "8000 spectators", "3": "20000 spectators", "4": "40000 spectators", "5": "80000 spectators"}),
    ("data_migration_size", 4, "We migrated nearly 800 GB of archives last weekend.", 800, "GB",
     {"1": "50 GB", "2": "200 GB", "3": "400 GB", "4": "800 GB", "5": "2 TB"}),
    ("festival_crowd", 4, "Almost 50000 devotees visited the temple on the festival day.", 50000, "people",
     {"1": "2000 people", "2": "10000 people", "3": "25000 people", "4": "50000 people", "5": "200000 people"}),
    ("home_loan_amount", 4, "Our home loan is about 4000000 rupees.", 4000000, "INR",
     {"1": "200000 INR", "2": "800000 INR", "3": "2000000 INR", "4": "4000000 INR", "5": "10000000 INR"}),
    ("highway_distance", 4, "The highway stretch under repair is about 60 km long.", 60, "km",
     {"1": "5 km", "2": "15 km", "3": "30 km", "4": "60 km", "5": "150 km"}),
    ("semester_fee", 4, "The semester fee is nearly 90000 rupees.", 90000, "INR",
     {"1": "5000 INR", "2": "20000 INR", "3": "45000 INR", "4": "90000 INR", "5": "200000 INR"}),
    ("office_headcount", 4, "Our office floor seats roughly 250 employees.", 250, "employees",
     {"1": "15 employees", "2": "60 employees", "3": "120 employees", "4": "250 employees", "5": "600 employees"}),
    ("train_delay", 4, "The train was delayed by almost 120 minutes yesterday.", 120, "minutes",
     {"1": "10 minutes", "2": "30 minutes", "3": "60 minutes", "4": "120 minutes", "5": "300 minutes"}),
    # ---- level 5 : need 8
    ("city_population", 5, "Our city has over 8000000 residents now.", 8000000, "residents",
     {"1": "100000 residents", "2": "500000 residents", "3": "2000000 residents", "4": "5000000 residents", "5": "8000000 residents"}),
    ("dam_capacity", 5, "The reservoir stores about 500000000 litres at full level.", 500000000, "litres",
     {"1": "5000000 litres", "2": "25000000 litres", "3": "100000000 litres", "4": "300000000 litres", "5": "500000000 litres"}),
    ("annual_turnover", 5, "The company's annual turnover crossed 50000000 rupees.", 50000000, "INR",
     {"1": "500000 INR", "2": "2000000 INR", "3": "10000000 INR", "4": "25000000 INR", "5": "50000000 INR"}),
    ("national_park_area", 5, "The national park spans roughly 900 square km.", 900, "sqkm",
     {"1": "20 sqkm", "2": "80 sqkm", "3": "250 sqkm", "4": "500 sqkm", "5": "900 sqkm"}),
    ("marathon_runners", 5, "Nearly 60000 runners joined the city marathon.", 60000, "runners",
     {"1": "1000 runners", "2": "5000 runners", "3": "15000 runners", "4": "30000 runners", "5": "60000 runners"}),
    ("server_storage", 5, "The new cluster provides 5 PB of usable storage.", 5, "PB",
     {"1": "50 TB", "2": "200 TB", "3": "800 TB", "4": "2 PB", "5": "5 PB"}),
    ("flight_distance", 5, "The direct flight covers about 13000 km.", 13000, "km",
     {"1": "500 km", "2": "2000 km", "3": "5000 km", "4": "9000 km", "5": "13000 km"}),
    ("leave_balance", 5, "I have accumulated 240 days of earned leave.", 240, "days",
     {"1": "5 days", "2": "20 days", "3": "60 days", "4": "120 days", "5": "240 days"}),
]

# ------------------------------------------------------------------ neg ---
# (attribute, text, underlying true_value). Text carries the negation scope;
# true_value is the proposition value BEFORE negation wording is applied,
# matching v0 convention (e.g. "not allergic" -> false).
NEG_NEW = [
    # false underlying (negated text denies something false) : need 20
    ("lactose_intolerance", "I am not lactose intolerant.", False),
    ("parking_available", "Parking is not available in our lane.", False),
    ("lift_working", "The lift is not working this morning.", False),
    ("canteen_open_sunday", "The canteen is not open on Sundays.", False),
    ("bus_route_changed", "My bus route has not changed this month.", False),
    ("fee_hiked", "The school fee has not been hiked this year.", False),
    ("meeting_cancelled", "Tomorrow's review meeting is not cancelled.", False),
    ("water_tanker_ordered", "We have not ordered a water tanker today.", False),
    ("project_delayed", "Our release is not delayed beyond Friday.", False),
    ("doctor_available_monday", "The doctor is not available on Monday morning.", False),
    ("wifi_down", "The office wifi is not down right now.", False),
    ("exam_postponed", "The entrance exam has not been postponed.", False),
    ("salary_delayed", "This month's salary was not delayed.", False),
    ("train_cancelled", "The 6 AM train is not cancelled today.", False),
    ("shop_closed", "The corner shop is not closed for renovation.", False),
    ("rain_expected", "No rain is expected in the city tonight.", False),
    ("power_cut_scheduled", "No power cut is scheduled for our block.", False),
    ("pet_allowed", "Pets are not allowed inside the clubhouse.", False),
    ("smoking_allowed", "Smoking is not allowed on this floor.", False),
    ("outside_food_allowed", "Outside food is not allowed in the theatre.", False),
    # true underlying (affirmative or true-negation) : need 16
    ("id_verified", "My ID was verified at the gate.", True),
    ("tickets_confirmed", "Our bus tickets are confirmed for Sunday.", True),
    ("report_submitted", "The weekly report was submitted on time.", True),
    ("keys_found", "I found the missing keys under the sofa.", True),
    ("parcel_received", "We received the parcel yesterday evening.", True),
    ("leave_approved", "My sick leave was approved without questions.", True),
    ("invoice_paid", "The vendor invoice has been paid in full.", True),
    ("phone_repaired", "My phone was repaired within two hours.", True),
    ("gate_locked", "The main gate was locked before midnight.", True),
    ("medication_taken", "I took my morning medication after breakfast.", True),
    ("backup_completed", "Last night's backup completed without errors.", True),
    ("form_accepted", "My application form was accepted at the counter.", True),
    ("delivery_completed", "The grocery delivery arrived before noon.", True),
    ("meeting_over", "The standup meeting is already over.", True),
    ("lights_off", "The corridor lights were off after 11 PM.", True),
    ("documents_signed", "All three documents were signed yesterday.", True),
]


def rows_for(category, prefix, entries, start_idx):
    rows = []
    for i, entry in enumerate(entries, start=start_idx):
        if category == "numeric_magnitude":
            attr, level, text, value, unit, scale = entry
            rows.append({
                "id": f"{prefix}_{i:03d}", "category": category,
                "original_text": text, "attribute": attr,
                "true_level": level, "true_value": value, "unit": unit,
                "scale": scale, "source": SRC, "source_reference": REF,
            })
        elif category == "polarity_negation":
            attr, text, value = entry
            rows.append({
                "id": f"{prefix}_{i:03d}", "category": category,
                "original_text": text, "attribute": attr,
                "true_value": value, "source": SRC, "source_reference": REF,
            })
        else:
            attr, level, text, scale = entry
            rows.append({
                "id": f"{prefix}_{i:03d}", "category": category,
                "original_text": text, "attribute": attr,
                "true_level": level, "scale": scale,
                "source": SRC, "source_reference": REF,
            })
    return rows


def main():
    v0 = [json.loads(line) for line in V0_PATH.open(encoding="utf-8") if line.strip()]
    assert len(v0) == 20, f"expected 20 v0 rows, got {len(v0)}"
    new_rows = []
    new_rows += rows_for("preference_intensity", "pref", PREF_NEW, 5)
    new_rows += rows_for("epistemic_certainty", "cert", CERT_NEW, 5)
    new_rows += rows_for("obligation_strength", "obl", OBL_NEW, 5)
    new_rows += rows_for("numeric_magnitude", "num", NUM_NEW, 5)
    new_rows += rows_for("polarity_negation", "neg", NEG_NEW, 5)
    assert len(new_rows) == 180, len(new_rows)

    # id uniqueness + text uniqueness
    ids = [r["id"] for r in v0] + [r["id"] for r in new_rows]
    assert len(set(ids)) == 200, "duplicate ids"
    texts = [r["original_text"] for r in v0 + new_rows]
    assert len(set(texts)) == 200, "duplicate texts"

    V1_PATH.parent.mkdir(parents=True, exist_ok=True)
    with V1_PATH.open("w", encoding="utf-8") as f:
        for r in v0 + new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    all_rows = v0 + new_rows
    for cat in ["preference_intensity", "epistemic_certainty", "obligation_strength", "numeric_magnitude"]:
        print(cat, sorted(Counter(r["true_level"] for r in all_rows if r["category"] == cat).items()))
    print("polarity_negation",
          Counter(str(r["true_value"]) for r in all_rows if r["category"] == "polarity_negation"))
    print(f"wrote {len(all_rows)} rows -> {V1_PATH}")


if __name__ == "__main__":
    main()
