package de.ovgu.ir.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class FieldNames {
	public final static String _ImagePath = "imagepath";
	public final static String _FilePath = "filepath";
	public final static String _FileName = "filename";
	public final static String _Content = "content";
	public final static String _Title = "title";
	public final static String _Date = "date";
	public final static String _Summary = "summary";
	public final static String _LastModifiedDate = "lastmodified";
	public final static String[] _QueryFieldNames = {_Content, _Title, _Date, _Summary};
}
