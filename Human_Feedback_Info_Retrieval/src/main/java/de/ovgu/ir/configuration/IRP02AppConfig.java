package de.ovgu.ir.configuration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Configuration
public class IRP02AppConfig {
	//@Value("${ir.indexpath}")
	String indexpath;
}
