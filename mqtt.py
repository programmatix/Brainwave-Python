if args.mqtt_url:
    mqtt_client = mqtt.Client()
    if args.mqtt_username and args.mqtt_password:
        mqtt_client.username_pw_set(args.mqtt_username, args.mqtt_password)
    mqtt_client.connect(args.mqtt_url)
    mqtt_client.loop_start()
    logger.info(f"Connected to MQTT on {args.mqtt_url}")

    # if mqtt_client:
    #     mqtt_client.publish('brainwave/eeg', json.dumps([
    #         {
    #             'channel': channel.channel_name,
    #             'delta': channel.band_powers.delta,
    #             'theta': channel.band_powers.theta,
    #             'alpha': channel.band_powers.alpha,
    #             'beta': channel.band_powers.beta,
    #             'gamma': channel.band_powers.gamma
    #         } for channel in eeg_data
    #     ]))

