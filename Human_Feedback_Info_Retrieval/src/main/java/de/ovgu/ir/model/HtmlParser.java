package de.ovgu.ir.model;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class HtmlParser extends StandardParser {
	@Override
	public ArrayList<Field> parse(File file) throws IOException {
		ArrayList<Field> fields = new ArrayList<Field>();
		fields.addAll(super.parse(file));
		
		Document htmlFile = Jsoup.parse(file, "utf-8");
		
		try {
			
			String title = htmlFile.title();
			fields.add(new TextField(FieldNames._Title, title, Field.Store.YES));
			
			String body = htmlFile.body().text();
			fields.add(new TextField(FieldNames._Content, body, Field.Store.YES));
			
			String summary = htmlFile.getElementsByTag("summary").text();
			fields.add(new TextField(FieldNames._Summary, summary, Field.Store.YES));
			
			Elements times = htmlFile.getElementsByTag("time");
			Iterator<Element> itr = times.iterator();
			String dateStr = "";
			while(itr.hasNext()) {
				Element time = itr.next();
				if(time.hasAttr("datetime")) {
					dateStr += time.getElementsByAttribute("datetime").attr("datetime") + " ";
				}
				else {
					dateStr += time.text() + " ";
				}
			}
			fields.add(new TextField(FieldNames._Date, dateStr + " ", Field.Store.YES));
			
		} catch(Exception e) {
			e.printStackTrace();
		}

		return fields;
	}
}
