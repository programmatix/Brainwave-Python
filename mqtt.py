if args.mqtt_url:
    mqtt_client = mqtt.Client()
    if args.mqtt_username and args.mqtt_password:
        mqtt_client.username_pw_set(args.mqtt_username, args.mqtt_password)
    mqtt_client.connect(args.mqtt_url)
    mqtt_client.loop_start()
    logger.info(f"Connected to MQTT on {args.mqtt_url}")
