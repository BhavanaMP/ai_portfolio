package de.ovgu.ir.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class QueryData {
	private String rank;
	private String imagepath;
	private String totalHits;
	private String score;
	private String title;
	private String filepath;
	private String ratings;
}
