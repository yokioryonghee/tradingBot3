with col2_date:
    default_end_date = datetime.datetime.strptime(END_DATE, '%Y-%m-%d').date()
    today = datetime.date.today()
    if default_end_date > today:
        default_end_date = today  # 미래면 오늘로 제한

    end_date = st.date_input(
        "End Date",
        value=default_end_date,
        min_value=datetime.date(2015, 12, 31),  # 필요시 조정
        max_value=today
    )
