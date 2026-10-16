import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False
# 解决中文乱码，图表全部中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
import shap

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="农业种植决策智能体", layout="wide")
st.title("🌾 农业种植决策智能体")

# ===================== 1. 作物数据库 =====================
crop_database = {
    "水稻_常规稻": {
        "seed": [30.0, 60.0],
        "fertilizer": [120.0, 180.0],
        "pesticide": [20.0, 40.0],
        "labor": [80.0, 140.0],
        "land": [50.0, 100.0],
        "price": 2.8,
        "base_yield": 550,
        "drought_coeff": 110,
        "pest_coeff": 110,
        "rain_opt": 950
    },
    "水稻_杂交稻": {
        "seed": [60.0, 110.0],
        "fertilizer": [120.0, 180.0],
        "pesticide": [20.0, 40.0],
        "labor": [80.0, 140.0],
        "land": [50.0, 100.0],
        "price": 3.2,
        "base_yield": 600,
        "drought_coeff": 100,
        "pest_coeff": 120,
        "rain_opt": 920
    },
    "玉米_普通玉米": {
        "seed": [40.0, 70.0],
        "fertilizer": [100.0, 160.0],
        "pesticide": [25.0, 45.0],
        "labor": [70.0, 120.0],
        "land": [40.0, 90.0],
        "price": 2.4,
        "base_yield": 520,
        "drought_coeff": 130,
        "pest_coeff": 85,
        "rain_opt": 800
    },
    "玉米_甜玉米": {
        "seed": [80.0, 130.0],
        "fertilizer": [110.0, 170.0],
        "pesticide": [30.0, 50.0],
        "labor": [90.0, 140.0],
        "land": [60.0, 110.0],
        "price": 3.6,
        "base_yield": 480,
        "drought_coeff": 140,
        "pest_coeff": 90,
        "rain_opt": 750
    }
}

# ===================== 2. 扩充后的四川地区气象&风险数据库 =====================
region_database = {
    "成都温江": {"drought": 0.10, "pest": 0.18, "rain": 960, "temp_sum": 4800},
    "成都都江堰": {"drought": 0.08, "pest": 0.15, "rain": 1100, "temp_sum": 4500},
    "成都郫都": {"drought": 0.12, "pest": 0.17, "rain": 940, "temp_sum": 4750},
    "成都崇州": {"drought": 0.11, "pest": 0.17, "rain": 950, "temp_sum": 4780},
    "绵阳": {"drought": 0.35, "pest": 0.20, "rain": 820, "temp_sum": 4600},
    "江油": {"drought": 0.33, "pest": 0.19, "rain": 840, "temp_sum": 4650},
    "南充": {"drought": 0.40, "pest": 0.25, "rain": 780, "temp_sum": 4900},
    "阆中": {"drought": 0.42, "pest": 0.26, "rain": 760, "temp_sum": 4950},
    "乐山": {"drought": 0.10, "pest": 0.18, "rain": 1020, "temp_sum": 5100},
    "眉山": {"drought": 0.15, "pest": 0.19, "rain": 910, "temp_sum": 4850},
    "雅安": {"drought": 0.05, "pest": 0.22, "rain": 1600, "temp_sum": 4400},
    "广元": {"drought": 0.45, "pest": 0.24, "rain": 700, "temp_sum": 4450},
    "达州": {"drought": 0.38, "pest": 0.27, "rain": 800, "temp_sum": 5000},
    "巴中": {"drought": 0.41, "pest": 0.23, "rain": 750, "temp_sum": 4700},
    "资阳": {"drought": 0.30, "pest": 0.21, "rain": 850, "temp_sum": 4880},
    "内江": {"drought": 0.32, "pest": 0.22, "rain": 830, "temp_sum": 4920},
    "自贡": {"drought": 0.34, "pest": 0.23, "rain": 810, "temp_sum": 4950},
    "泸州": {"drought": 0.20, "pest": 0.24, "rain": 1050, "temp_sum": 5150}
}

# ===================== 3. 核心产量收益计算函数 =====================
def calculate_result(area, seed, fer, pest, labor, land, drought, pest_risk, rain, temp_sum, price, crop_name):
    crop_info = crop_database[crop_name]
    base_yield = crop_info["base_yield"]
    drought_coeff = crop_info["drought_coeff"]
    pest_coeff = crop_info["pest_coeff"]
    rain_opt = crop_info["rain_opt"]

    # 干旱、虫害减产
    yield_val = base_yield - drought * drought_coeff - pest_risk * pest_coeff
    # 降雨偏离最优值减产
    rain_loss = abs(rain - rain_opt) * 0.08
    yield_val -= rain_loss
    # 积温不足减产
    if temp_sum < 4500:
        yield_val -= (4500 - temp_sum) * 0.04
    # 农资投入影响
    yield_val = yield_val - seed * 0.11 - fer * 0.16
    yield_val = max(yield_val, 120)

    cost_per_mu = seed + fer + pest + labor + land
    total_cost = cost_per_mu * area
    total_output = yield_val * area * price
    net_income = total_output - total_cost
    return round(yield_val, 2), round(net_income, 2), round(cost_per_mu, 2)

# ===================== 4. 多目标寻优函数 =====================
def multi_object_optimize(area, seed, fer, pest, labor, land, drought, pest_risk, rain, temp_sum, price, crop_name, obj_target):
    candidates = []
    for a_ratio in np.linspace(0.5, 1.6, 25):
        for f_ratio in np.linspace(0.7, 1.3, 20):
            new_area = area * a_ratio
            new_fer = fer * f_ratio
            y, inc, c = calculate_result(new_area, seed, new_fer, pest, labor, land, drought, pest_risk, rain, temp_sum, price, crop_name)
            candidates.append({"面积":new_area, "肥料":new_fer, "亩产":y, "净收益":inc, "亩均成本":c})
    df_cand = pd.DataFrame(candidates)
    if obj_target == "收益优先":
        best = df_cand.loc[df_cand["净收益"].idxmax()]
    elif obj_target == "产量优先":
        best = df_cand.loc[df_cand["亩产"].idxmax()]
    else:
        best = df_cand.loc[(df_cand["亩产"] - df_cand["净收益"]*0.01).idxmax()]
    return best, df_cand

# ===================== 5. 动态农事建议（升级：客观风险 + 主观调控方案） =====================
def get_farm_advice(drought, pest_risk, fer, crop_name):
    advice_list = []
    # 干旱风险（客观不可变）+ 人为应对措施
    if drought > 0.3:
        advice_list.append("⚠️客观条件：干旱风险偏高，自然降水不足，容易造成作物缺水减产。👉主观调控：优先选用耐旱品种；配套灌溉设施；增施有机肥保水；干旱期可喷施抗旱调节剂，减少水分蒸腾损耗。")
    elif drought > 0.2:
        advice_list.append("ℹ️客观条件：存在轻度干旱风险，旱季土壤墒情不足。👉主观调控：做好田间蓄水保墒，覆盖秸秆减少蒸发，必要时少量补水，避免作物受旱。")
    else:
        advice_list.append("✅客观条件：干旱风险较低，雨水条件良好。👉主观调控：维持常规水分管理即可，重点做好雨季排涝。")

    # 虫害风险（客观不可变）+ 农药/防控方案
    if pest_risk > 0.22:
        advice_list.append("⚠️客观条件：地块虫害发生概率高，容易爆发虫害造成减产。👉主观调控：优先物理+生物绿色防控；虫害高发期提前预防性喷施低毒杀虫药剂；定期巡查田间，尽早清除虫卵。")
    elif pest_risk > 0.15:
        advice_list.append("ℹ️客观条件：虫害中等，存在虫害爆发可能性。👉主观调控：定期巡查田间，及时清除杂草虫源；监测虫口数量，达到阈值时使用对应农药防治。")
    else:
        advice_list.append("✅客观条件：虫害风险较低。👉主观调控：维持常规田间监测，以预防为主，减少农药使用。")

    # 肥料投入（人为可调）
    crop_info = crop_database[crop_name]
    fer_low = crop_info["fertilizer"][0]
    fer_high = crop_info["fertilizer"][1]
    if fer > fer_high:
        advice_list.append("ℹ️当前肥料投入偏高，存在边际收益递减，农资成本浪费。👉主观调控：适当减少化肥用量，配合有机肥，避免过量施肥造成徒长、倒伏。")
    elif fer < fer_low:
        advice_list.append("ℹ️当前肥料投入偏低，养分不足会限制产量。👉主观调控：适度增施氮磷钾复合肥，作物关键生长期补充叶面肥，提升长势。")
    else:
        advice_list.append("✅肥料投入处于合理区间。👉主观调控：可以分阶段施肥，基肥+追肥搭配，提高肥料利用率。")
    return "\n".join(advice_list)

# ===================== 6. SHAP函数 + SHAP中文解读 =====================
def get_shap_result(area,seed,fer,pest,labor,land,drought,pest_risk,rain,temp_sum,crop_name):
    X = pd.DataFrame([[area,seed,fer,pest,labor,land,drought,pest_risk,rain,temp_sum]],
        columns=["种植面积","种子成本","肥料成本","农药成本","人工成本","土地成本","干旱风险","虫害风险","降雨量","积温"])
    def pred_func(data):
        res = []
        for row in data:
            y,_,_ = calculate_result(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],crop_database[crop_name]["price"],crop_name)
            res.append(y)
        return np.array(res)
    explainer = shap.KernelExplainer(pred_func, np.zeros((1,10)))
    shap_values = explainer.shap_values(X)
    return X, shap_values

def shap_text_interpret(X,shap_values):
    vals = shap_values[0]
    cols = list(X.columns)
    pair = list(zip(cols, vals))
    pair_sorted = sorted(pair, key=lambda x:abs(x[1]), reverse=True)
    top3 = pair_sorted[:3]
    text = "SHAP可解释分析：当前影响产量最主要的三个因素："
    for name,val in top3:
        if val>0:
            text += f"{name}(正向提高产量)，"
        else:
            text += f"{name}(降低产量)，"
    text = text[:-1]+"。\n建议优先管控负向影响因素，合理调整各项种植投入。"
    return text

# ===================== 7. 问答智能体 =====================
def agent_chat(question, area, yield_res, inc_res, drought, pest_risk):
    q = question.lower()
    if "为什么" in q and "推荐" in q:
        return f"AI推荐方案是综合产量、成本、灾害风险算出。当前干旱风险{drought}、虫害风险{pest_risk}会压低产量；扩大种植面积会增加总产量，但土地人工成本同步上涨，因此不是面积越大越好。"
    elif "收益" in q or "赚" in q:
        return f"当前预估总净收益：{inc_res}元，收益受产量、各项农资成本、粮食价格共同影响。"
    elif "产量" in q:
        return f"当前预估亩产：{yield_res}公斤，产量受天气、病虫害、肥料投入共同决定。"
    elif "风险" in q:
        return f"当前干旱风险{drought}，虫害风险{pest_risk}，风险越高，减产可能性越大。"
    else:
        return "我可以帮你分析产量、收益、风险、方案相关问题，请继续提问。"

# ===================== 页面布局 =====================
col_left, col_right = st.columns([1,2])

with col_left:
    st.subheader("📋 参数录入面板")
    crop_name = st.selectbox("选择作物品种", list(crop_database.keys()))
    region_name = st.selectbox("选择地区", list(region_database.keys()))

    region_info = region_database[region_name]
    # 自动带出地区数据，可手动修改
    drought = st.number_input("干旱风险（可手动修改）", min_value=0.0, max_value=1.0, value=region_info["drought"], step=0.01)
    pest_risk = st.number_input("虫害风险（可手动修改）", min_value=0.0, max_value=1.0, value=region_info["pest"], step=0.01)
    rain = st.number_input("年降雨量（毫米）", value=region_info["rain"])
    temp_sum = st.number_input("年积温", value=region_info["temp_sum"])

    # ========= 新增：四合一参数对照表（折叠面板） =========
    st.divider()
    with st.expander("❓ 四个参数怎么填？看这张对照表"):
        help_table = pd.DataFrame([
            {"参数":"干旱风险","数值范围":"0~0.15","程度":"低风险","说明":"雨水充足，几乎不担心缺水"},
            {"参数":"干旱风险","数值范围":"0.16~0.30","程度":"中低风险","说明":"偶尔干旱，一般影响不大"},
            {"参数":"干旱风险","数值范围":"0.31~0.45","程度":"中高风险","说明":"经常缺水，会明显减产"},
            {"参数":"干旱风险","数值范围":"0.46~1.00","程度":"高风险","说明":"严重缺水，几乎无收"},
            {"参数":"虫害风险","数值范围":"0~0.15","程度":"低风险","说明":"虫害少，常规防治即可"},
            {"参数":"虫害风险","数值范围":"0.16~0.25","程度":"中低风险","说明":"少量虫害，及时防治没事"},
            {"参数":"虫害风险","数值范围":"0.26~0.40","程度":"中高风险","说明":"虫害常见，会减产"},
            {"参数":"虫害风险","数值范围":"0.41~1.00","程度":"高风险","说明":"虫害严重，重点防治"},
            {"参数":"年降水量","数值范围":"<700mm","程度":"偏少","说明":"偏干旱，需灌溉补墒"},
            {"参数":"年降水量","数值范围":"700~1000mm","程度":"适中","说明":"比较适合多数作物"},
            {"参数":"年降水量","数值范围":">1000mm","程度":"偏多","说明":"雨水多，注意防涝"},
            {"参数":"年积温","数值范围":"<4500℃","程度":"偏低","说明":"生长季偏短，产量受限"},
            {"参数":"年积温","数值范围":"4500~5000℃","程度":"适中","说明":"适合水稻玉米等大田作物"},
            {"参数":"年积温", "数值范围":">5000℃","程度":"充足","说明":"热量足，作物长得快"},
        ])
        st.dataframe(help_table, use_container_width=True)
        st.caption("提示：选择地区后系统会自动带出参考值，你可根据当年实际情况手动修改。")
    # =====================================================

    crop_info = crop_database[crop_name]
    st.info(f"参考成本区间：种子{crop_info['seed'][0]}-{crop_info['seed'][1]}，肥料{crop_info['fertilizer'][0]}-{crop_info['fertilizer'][1]}，基准单价：{crop_info['price']}元/公斤")
    area = st.number_input("种植面积(亩)", min_value=0.1, value=10.0, step=0.5)
    seed = st.number_input("种子成本 元/亩", min_value=0.0, value=np.mean(crop_info["seed"]))
    fer = st.number_input("肥料成本 元/亩", min_value=0.0, value=np.mean(crop_info["fertilizer"]))
    pest_cost = st.number_input("农药成本 元/亩", min_value=0.0, value=np.mean(crop_info["pesticide"]))
    labor = st.number_input("人工成本 元/亩", min_value=0.0, value=np.mean(crop_info["labor"]))
    land = st.number_input("土地成本 元/亩", min_value=0.0, value=np.mean(crop_info["land"]))

    obj_target = st.selectbox("优化目标", ["收益优先","产量优先","风险优先"])
    # 粮价情景模拟
    price_mode = st.selectbox("粮食价格情景", ["基准价","下跌10%","下跌20%","上涨10%"])
    base_price = crop_info["price"]
    if price_mode == "下跌10%":
        use_price = base_price * 0.9
    elif price_mode == "下跌20%":
        use_price = base_price * 0.8
    elif price_mode == "上涨10%":
        use_price = base_price * 1.1
    else:
        use_price = base_price

with col_right:
    st.subheader("📊 AI预测 & 智能寻优")
    yield_res, net_inc, cost_mu = calculate_result(area, seed, fer, pest_cost, labor, land, drought, pest_risk, rain, temp_sum, use_price, crop_name)
    best_sol, cand_df = multi_object_optimize(area, seed, fer, pest_cost, labor, land, drought, pest_risk, rain, temp_sum, use_price, crop_name, obj_target)

    c1,c2,c3 = st.columns(3)
    c1.metric("预估亩产（公斤）", yield_res)
    c2.metric("总净收益（元）", net_inc)
    c3.metric("亩均成本（元）", cost_mu)

    # 判断当前方案是否最优
    diff_inc = best_sol["净收益"] - net_inc
    if diff_inc < 80:
        st.success("✅ 当前种植方案已经接近最优区间，暂无更优改良方案！建议做好田间管护。")
    else:
        st.success(f"📌 AI寻优完成！推荐方案：面积{best_sol['面积']:.1f}亩，肥料{best_sol['肥料']:.1f}元/亩，预估亩产{best_sol['亩产']:.2f}公斤，净收益{best_sol['净收益']:.2f}元。")
        compare_df = pd.DataFrame([
            {"方案类型":"当前方案","面积(亩)":area,"亩产(公斤)":yield_res,"亩均成本":cost_mu,"净收益":net_inc},
            {"方案类型":"AI最优方案","面积(亩)":best_sol["面积"],"亩产(公斤)":best_sol["亩产"],"亩均成本":best_sol["亩均成本"],"净收益":best_sol["净收益"]}
        ])
        st.dataframe(compare_df, use_container_width=True)
        # 柱状对比图【全部中文标题】
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
        ax1.bar(compare_df["方案类型"], compare_df["亩产(公斤)"], color=["#74b9ff","#00b894"])
        ax1.set_title("亩产对比")
        ax1.set_ylabel("亩产（公斤）")
        ax2.bar(compare_df["方案类型"], compare_df["净收益"], color=["#74b9ff","#00b894"])
        ax2.set_title("净收益对比")
        ax2.set_ylabel("净收益（元）")
        plt.tight_layout()
        st.pyplot(fig)

    # 动态农事建议
    st.subheader("🌱 动态农事指导建议")
    farm_advice = get_farm_advice(drought, pest_risk, fer, crop_name)
    st.write(farm_advice)

    # SHAP + 自动文字解读
    st.subheader("🔍 SHAP模型可解释分析")
    X_shap, shap_val = get_shap_result(area,seed,fer,pest_cost,labor,land,drought,pest_risk,rain,temp_sum,crop_name)
    fig2, ax3 = plt.subplots(figsize=(9,4))
    shap.summary_plot(shap_val, X_shap, plot_type="bar", show=False)
    ax3.set_xlabel("因素对产量影响大小")
    st.pyplot(fig2)
    shap_text = shap_text_interpret(X_shap, shap_val)
    st.info(shap_text)

# ========== 底部问答智能体 ==========
st.divider()
st.subheader("🤖 种植问答智能体")
user_q = st.text_input("向智能体提问（产量、收益、风险、方案）")
if user_q:
    ans = agent_chat(user_q, area, yield_res, net_inc, drought, pest_risk)
    st.info(ans)
